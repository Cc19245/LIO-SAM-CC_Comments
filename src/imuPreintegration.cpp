/**
 * @file imuPreintegration.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-31
 * 
 * 1.订阅IMU原始测量数据，并进行预积分得到高频的、纯里程计的、有全局漂移的位姿
 * 2.订阅后端LiDAR优化的增量里程计位姿(有全局漂移)，并加入因子图中优化，从而矫正IMU的零偏
 * 3.订阅后端LiDAR优化的全局里程计位姿(无全局漂移)，并把IMU高频里程计位姿的增量叠加到这个全局位姿上，
 *   得到最终IMU预测的高频的、无全局漂移的里程计位姿
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "../include/utility.h"  

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

//; 这些符号代表了整体的状态变量，即所有帧的P V Q ba bg
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    ros::Subscriber subImuOdometry;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;

    Eigen::Affine3f lidarOdomAffine;
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener;
    tf::StampedTransform lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::Odometry> imuOdomQueue;

    TransformFusion()
    {
        // 如果lidar帧和baselink帧不是同一个坐标系
        // 通常baselink指车体系
        if (lidarFrame != baselinkFrame)
        {
            try
            {
                // 查询一下lidar和baselink之间的tf变换,ros::Time(0)代表查询最新时刻的坐标变换
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                //; lidar2Baselink = T_lidar_baselink
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s", ex.what());
            }
        }

        //; 订阅后端优化的全局位姿，注意是带回环检测的全局位姿，没有累计漂移
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, 
            &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        
        //; 订阅IMU预积分节点发出的里程计位姿，但是它不是全局位姿，因此是有累计漂移的
        subImuOdometry = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000,
            &TransformFusion::imuOdometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        pubImuPath = nh.advertise<nav_msgs::Path>("lio_sam/imu/path", 1);
    }

    // ros数据格式转化成Eigen数据格式
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);   //; 因为pcl也是依赖于eigen的，所以这里输出结果是eigen的数据类型
    }

    // 将全局位姿保存下来, 其实就是赋值到类的成员变量中
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        lidarOdomAffine = odom2affine(*odomMsg);
        lidarOdomTime = odomMsg->header.stamp.toSec();
    }

    
    /**
     * @brief 利用IMU预积分节点得到的高频里程计位姿(有漂移)，计算位姿增量；
     *  然后把位姿增量叠加到后端优化发来的全局位姿(无漂移)上，得到全局无漂移的高频里程计位姿
     * @param[in] odomMsg 
     */
    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
    {
        // static tf
        static tf::TransformBroadcaster tfMap2Odom;
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        // 发送静态tf，odom系和map系将他们重合
        //; 在ros中分为map系和odom系，其中map系就是地图的坐标系，而odom坐标系则是机器人上电时刻那个初始位置所在的坐标系
        //; 这种在基于已知地图来进行定位导航的应用中是不同的。但是这里由于是SLAM，所以map系和odom系就是一样的
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);
        // imu得到的里程记结果送入这个队列中
        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        // 如果没有收到lidar位姿就return 
        //; 如果还没有收到带回环的全局一致的里程计位姿，那么也不用往它上面进行补偿
        if (lidarOdomTime == -1)
            return;
        // 弹出时间戳小于最新lidar位姿时刻之前的imu里程记数据
        while (!imuOdomQueue.empty())
        { 
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        // 计算最新队列里imu里程记的增量
        //; IMU频率的位姿是和IMU频率同步的，所以之前弹出了lidar里程计时间戳之前的IMU位姿后，
        //; imuOdomQueue最前面的就是lidar里程计时间戳对应的IMU位姿，最后面的就是最新的IMU频率位姿
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        // 增量补偿到lidar的位姿上去，就得到了最新的预测的位姿
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        float x, y, z, roll, pitch, yaw;
        // 分解成平移+欧拉角的形式
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);

        // publish latest odometry
        // 发送全局一致位姿的最新位姿，即IMU频率的位姿
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        // 更新tf
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);  //; 里程计位姿转成tf消息
        if (lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;   //; T_w_b = T_w_l * T_l_b
        // 更新odom到baselink的tf
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        // 发送imu里程记的轨迹
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        // 控制一下更新频率，不超过10hz
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;  //; 发送的是IMU频率的全局一致的位姿
            // 将最新的位姿送入轨迹中
            imuPath.poses.push_back(pose_stamped);
            // 把lidar时间戳之前的轨迹全部擦除
            while (!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());

            // 发布轨迹，这个轨迹实际上是可视化imu预积分节点输出的预测值
            //; 看这个写法，还可以实时查询是否有定义这个话题的节点，然后再决定是否发布，这样可以节省资源
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};


// Step  IMU 预积分类
class IMUPreintegration : public ParamServer
{
public:
    std::mutex mtx;

    ros::Subscriber subImu;
    ros::Subscriber subOdometry;
    ros::Publisher pubImuOdometry;

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;

    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    //; 最新的优化后的位姿和bias
    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;

    //; T_imulidar_lidar，其中旋转一定是单位帧，因为这里的IMU是已经转换到和LiDAR坐标轴xyz完全平行的形式了，
    //; 所以只剩下旋转了。定义旋转到和LiDAR坐标轴完全平行、但是坐标系原点不变的IMU坐标系为imulidar, 那么这里
    //; 要的其实是T_imulidar_lidar。而配置文件中我们给的extTrans是T_lidar_imu的平移部分，即t_lidar_imu。
    //; 而由于作者的IMU安装和LiDAR恰好平行，所以imulidar = imu，即imu坐标系本身就和imulidar坐标系相同，因此
    //; T_imulidar_lidar = T_imu_lidar，则t_imu_lidar = -R_lidar_imu^T * t_lidar_imu = -t_lidar_imu
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    //; T_lidar_imulidar
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration()
    {
        //; 订阅imu消息
        //! 问题：传入this指针是干什么的？应该是说调用的回调函数是当前这个对象的函数？
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, 
            &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());

        //; 订阅后端优化发送来的Lidar里程计的消息，注意这里订阅的是 增量式的里程计位姿，
        //; 是不带回环检测的纯里程计，后端还会发布一个带回环检测结果的全局一致的位姿
        subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 5, 
                &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic + "_incremental", 2000);

        // Step 1 IMU预积分的协方差设置
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);     // gyro white noise in continuous
        p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);          // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());

        // Step 2 状态先验因子协方差设置
        // 初始位姿置信度设置比较高
        //; 变量一共是6维，每个维度的噪声都是0.01，都给列举出来了
        priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        // 初始速度置信度就设置差一些
        //; 变量一共是3位，每个维度噪声都是1e4，这里直接把三个维度的值都赋值成一个，而不是一一列举出来
        priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        // 零偏的置信度也设置高一些
        priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);                                                           // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());               // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

        // Step 3 新建两个预积分对象，一个用于预积分，一个用于odom优化
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization
    }

    void resetOptimization()
    {
        // gtsam初始化
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;    
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    // 订阅地图优化节点的增量里程记消息
    void odometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
    {
        //; 线程锁，基于变量的生命周期，也就是退出了这个回调函数之后会自动释放线程锁
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = ROS_TIME(odomMsg);  //; 消息的时间戳

        // make sure we have imu data to integrate
        // 确保imu队列中有数据
        if (imuQueOpt.empty())
            return;
        // 获取里程记位姿
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        // 该位姿是否出现退化，如果把covariance[0]=1的话那么这个Lidar里程计就有退化的风险，也就是精度会下降
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
        // 把位姿转成gtsam的格式
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

        // Step A: 系统没有初始化，则把第一帧位姿拿出来，加入因子图中初始化
        // 首先初始化系统
        if (systemInitialized == false)
        {
            // 优化问题进行复位，主要是对gtsam进行一些初始化设置
            resetOptimization();

            // pop old IMU message
            // 将这个里程记消息之前的imu信息全部扔掉
            //; 因为IMU只能形成帧间约束，所以第一帧lidar消息之前对应的IMU数据都是没用的
            while (!imuQueOpt.empty())
            {
                //; delta_t是一个const double，赋值是0，不知道这里写了是干什么的
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            // 将lidar的位姿转移到imu坐标系下
            //; T_odom_imulidar = T_odom_lidar * T_lidar_imulidar，其中odom就是world系
            prevPose_ = lidarPose.compose(lidar2Imu);  

            // Step 1 构建因子图
            // Step 1.1 给状态添加先验因子
            // 设置其初始位姿和置信度
            //; 先验因子，表示对某个状态变量的先验估计，约束状态变量不会离这个先验值过远，先验因子是有置信度的
            //; 给第0个位姿X(0)加入先验因子的约束，约束值是prevPose_，协方差矩阵是priorPoseNoise
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            // 添加先验因子到因子图中
            graphFactors.add(priorPose);

            // initial velocity
            // 初始化速度，这里就直接赋0了。一开始确实不知道是多少，因此协方差矩阵也很大，置信度很小
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            // 将对速度的约束也加入到因子图中
            graphFactors.add(priorVel);

            // initial bias
            // 初始化零偏
            prevBias_ = gtsam::imuBias::ConstantBias();  //; 关于零偏在gtsam中有专门的类定义，内部初始化都是0
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            // 零偏加入到因子图中
            graphFactors.add(priorBias);

            // Step 1.2 添加状态，即要优化的变量
            // 以上把约束加入完毕，下面开始添加状态量
            // add values
            // 将各个状态量赋成初始值
            //; 前面的都只是因子，也就是约束(图优化中的边)。但是对于节点，还要给节点一个优化前的初值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);

            // Step 1.3.更新进优化器，先优化一次
            // optimize once
            // 把约束和状态量更新进isam优化器
            optimizer.update(graphFactors, graphValues);

            // 进优化器之后保存约束和状态量的变量就清零
            graphFactors.resize(0);
            graphValues.clear();

            // 预积分的接口，使用初始零偏进行初始化
            //! 疑问：这里为什么要把零偏重新进行初始化为0？
            //; 解答：猜测是因为gtsam中IMU预积分的类，在执行完一次优化之后会把零偏补偿到预积分结果里面?
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

            key = 1;
            systemInitialized = true;
            return;
        }

        // reset graph for speed
        // Step B: 当正常运行一段时间后，因子图过大计算变慢，就清空因子图不再优化之前的值，重新添加新的因子图
        // 当isam优化器中加入了较多的约束后，为了避免运算时间变长，就直接清空因子图，然后重新构建
        //; 这里的操作和滤波有点相似，只不过滤波只保持最新帧的位姿，这里是保持100帧
        if (key == 100)
        {
            // get updated noise before reset
            // 取出最新时刻位姿 速度 零偏的协方差矩阵
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = 
                gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = 
                gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = 
                gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
            // reset graph
            // 复位整个优化问题
            resetOptimization();
            // add pose
            // 将最新的位姿，速度，零偏以及对应的协方差矩阵加入到因子图中
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }

        // Step C: 大部分情况，把IMU预积分和后端LiDAR位姿先验加入因子图中优化，从而更新IMU零偏
        // 将两帧之间的imu做积分
        // Step 1 取出这帧LiDAR之前的IMU数据，进行预积分
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            //; IMU形成的是第N帧和第N-1帧之间的帧间约束，因此要把当前lidar帧之前的所有IMU数据都取出来
            if (imuTime < currentCorrectionTime - delta_t)
            {
                // 计算两个imu量之间的时间差
                //; 如果是第一个IMU数据，那么lastImuT_opt还没有被更新过，初始赋值是负数，这里就把这个IMU数据
                //; 离上一次IMU数据的时间手动设置0.002。这种情况只会发生一次
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                // 调用预积分接口将imu数据送进去处理
                imuIntegratorOpt_->integrateMeasurement(
                    gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                    gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);
                // 记录当前imu时间
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }

        // add imu factor to graph
        // Step 2 把两帧LiDAR之间的IMU预积分约束加入因子图中
        // 两帧间imu预积分完成之后，就将其转换成预积分约束
        //; 指针转换，因为之前用的是计算预积分的类，为了把预积分加入到因子图中，需要转化成因子图可以识别的预积分因子类
        const gtsam::PreintegratedImuMeasurements &preint_imu = 
            dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
        // 预积分约束对相邻两帧之间的位姿 速度 零偏形成约束
        //! 问题：这里为什么没有B(key)？
        //; 解答：零偏后面添加了单独定义的因子
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);

        // add imu bias between factor
        // 零偏的约束，两帧间零偏相差不会太大，因此使用常量约束
        //; 1.BetweenFactor就是帧间约束，是通用的定义形式。而ImuFactor其实也属于BetweenFactor，
        //;   但是由于其计算比较复杂，所以gtsam中单独定义了IMUFactor这种因子
        //; 2.这里约束当前帧和上一帧的bias，由于bias是随机游走，所以相邻两帧时间很近不会发生变化，
        //;   因此是常量约束。但是随机游走会受到时间的影响，所以这里利用预积分类中保存的相邻两次的时间差
        //;   来加入随机游走的影响。
        //! 问题：随机游走到底是怎么影响的？当前帧和上一帧不是常量约束保持不变吗?
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
            B(key - 1), B(key), gtsam::imuBias::ConstantBias(), 
            gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        
        // add pose factor
        // Step 3 把后端优化的LiDAR位姿加入因子图中作为先验，就是靠它来优化IMU的零偏的
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        // lidar位姿补偿到imu坐标系下，同时根据是否退化选择不同的置信度，作为这一帧的先验估计
        //; 这里和论文中有点不一样，这里是把lidar里程计的结果作为一个先验约束，而论文中说的是帧间约束
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);

        // insert predicted values
        // Step 4 给因子图中要优化的变量赋初值
        //; 输入上一帧的位姿和bias，结合计算的当前帧和上一帧之间的IMU预积分，预测得到当前帧的位姿。
        //; 注意这里用的是IMU积分来预测状态，而不是使用LiDAR结果当做预测状态，本质上是因为因子图
        //; 中优化的状态变量类似VIO，是15维的；而LiDAR只能发来一个6维的位姿，所以并不够作为这里的预测
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        // 预测量作为初始值插入因子图中
        graphValues.insert(X(key), propState_.pose());  //! 这里可不可以直接使用lidar里程计的位姿？
        graphValues.insert(V(key), propState_.v());  //; 速度这个只能使用IMU推算的结果
        graphValues.insert(B(key), prevBias_);   //; 零偏直接使用上一次的零偏作为初值

        // optimize
        // Step 5 执行优化
        //! 下面调用两次update，是不是第一次有形参的只是把因子和变量加入因子图中，第二次没有形参的
        //! 才是真正执行优化？或者说下面就是update了两次，执行了两次优化？
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);  //; 因子和变量都清零，方便下一次添加
        graphValues.clear();

        // Overwrite the beginning of the preintegration for the next step.
        // Step 6 从因子图中取出优化后的最新状态，作为当前帧的最佳估计
        gtsam::Values result = optimizer.calculateEstimate(); 
        prevPose_ = result.at<gtsam::Pose3>(X(key));
        prevVel_ = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);   
        prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));

        // Reset the optimization preintegration object.
        // Step 7 把优化后的最新的零偏更新到预积分类中，为下次预积分做准备
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_); 

        // check optimization
        // 一个简单的失败检测
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();   // 状态异常就直接复位了
            return;
        }

        // after optiization, re-propagate imu odometry preintegration
        // Step 8 利用矫正的IMU零偏，更新IMU状态预测的预积分对象的预积分结果
        //; 因子图优化之后的最新的状态
        prevStateOdom = prevState_; 
        prevBiasOdom = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        // 首先把lidar帧之前的imu状态全部弹出去
        //; 注意这里使用的就是IMU状态推算的queue了，而不是上面优化使用的queue
        //; IMU :       **************
        //;                     |  使用这之后的IMU数据进行航迹推算，得到高频的IMU里程计
        //; 优化后的odom:        &(在这个时间戳之前的所有IMU数据弹出)    
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            // 这个预积分变量复位
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            // 然后把剩下的imu状态重新积分
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);
                //; 这里只是计算了预积分，还没有更新最新的IMU频率的位姿。
                //; 因为这个是odom的回调函数，更新IMU频率的位姿会在IMU的回调函数中实现
                imuIntegratorImu_->integrateMeasurement(
                    gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                    gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;

        // Step 9 IMU回调函数中的标志位置位，说明此时已经完成了一次优化，已经有了初始的IMU零偏，可以预积分预测状态了
        doneFirstOpt = true;  
    }


    bool failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        // 如果当前速度大于30m/s，108km/h就认为是异常状态，
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        // 如果零偏太大，那也不太正常
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }


    void imuHandler(const sensor_msgs::Imu::ConstPtr &imu_raw)
    {
        //;注意用的是lock_guard, 在这个变量声明周期结束之后就会自动释放线程锁
        std::lock_guard<std::mutex> lock(mtx);  

        // Step 1 IMU 测量值的坐标系转换，把IMU测量结果转到LiDAR坐标下，并得到LiDAR的姿态
        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        // Step 2: 存储IMU数据到队列中，为odom回调函数的使用做准备
        // 注意这里有两个imu的队列，作用不相同，一个用来执行预积分和位姿的优化，一个用来更新最新imu状态
        imuQueOpt.push_back(thisImu);   //; 用来执行预积分和位姿的优化
        imuQueImu.push_back(thisImu);   //; 用来更新最新imu状态

        //; 如果没有发生过优化就return，因为没有经过后端LiDAR位姿的矫正，不知道IMU零偏，预积分结果不准
        if (doneFirstOpt == false)
            return;

        //; 执行到这里的时候，说明doneFirstOpt是true，即在odom的回调函数中已经执行过后端优化了
        double imuTime = ROS_TIME(&thisImu);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);  //; 相邻两帧IMU时间
        lastImuT_imu = imuTime;

        // integrate this single imu message
        // Step 3: 把这个IMU数据加入预积分类中，计算预积分
        //; 1.下面这个预积分类是用于IMU频率的位姿推算的，这个预积分在odom回调中的最后也进行了调用
        //; 2.但是注意这里的预积分调用和odom回调中的预积分调用是不冲突的。假设第一次后端优化已经完成，
        //;   这时候每次进来的IMU数据都会调用下面的预积分进行计算，然后预测最新的IMU频率位姿。
        //;   如果此时再来一帧odom数据，后端优化又执行一次，此时bias会发生变化。由于是多线程，并且IMU
        //;   频率很高，所以在执行完这一次优化后，可能又进入了几次IMU的回调函数执行了下面的预积分，但实际上
        //;   由于优化后bias发生了变化，这几次预积分都是错误的，因此就在odom的回调函数中重新计算了这些预积分。
        //;   这样下次再进入IMU的回调函数执行下面的预积分的时候，就是在更正后的预积分的基础上继续进行预积分了。
        //! 问题：IMU和odom的回调函数都调用了imuIntegratorImu_变量执行预积分，不会发生线程冲突吗？
        imuIntegratorImu_->integrateMeasurement(
            gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
            gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z), dt);

        // predict odometry
        // Step 4: 根据上一次的状态和这次IMU预积分结果，预测最新的状态
        //; 输入上一帧的位姿和bias，结合计算的当前帧和上一帧之间的IMU预积分，预测得到当前帧的位姿
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        // Step 5: 发布IMU频率的里程计
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        // 将这个状态转到lidar坐标系下去发送出去
        //; 疑问：注意IMU数据在积分前已经被转到LiDAR坐标系下了，这里为什么还要把积分的位姿转到LiDAR系下？
        //; 解答：其实是因为之前IMU测量数据只有旋转，相当于是在以IMU原点为中心、坐标轴和LiDAR平行的坐标系下
        //;      积分得到位姿(这里称为imulidar坐标系)，而非真正的LiDAR坐标系，
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        //; T_odom_lidar = T_odom_imulidar * T_imulidar_lidar，其中odom系就是world系
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar); 

        //; 这里发布的是lidar的位姿
        //! 问题：这里存在一点小问题：odom发送过程中IMU的零偏是在变化的，在后端发来lidar的位姿进行优化之前
        //!      是一个零偏，后端发来lidar位姿优化之后变成了另一个零偏，但是此时前面的odom已经发送出去了，
        //!      不能更改了。所以我感觉这也是为什么imageProjection函数中使用odom进行平移畸变矫正的时候
        //!      用的是起始和结束位姿的差，而不是相邻两个odom的位姿进行插值，因为有两段计算公式不同的odom。
        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        //; 角速度和上次优化得到的角速度零偏相加，相当于得到此时估计的最优的角速度值
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "roboat_loam");

    IMUPreintegration ImuP;

    TransformFusion TF;

    //; 关于打印的颜色配置：https://blog.csdn.net/u014470361/article/details/81512330
    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");

    //; 多线程的触发，开了四个线程用于ros的回调，这样通过并发的方式可以使回调的速度得到提升
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin(); //; 和原来的spin一样，阻塞到这里，等待回调

    return 0;
}
