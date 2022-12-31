/**
 * @file imageProjection.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-31
 * 
 *  1. 从原始IMU数据中读取姿态，用于整个LIO系统初始化；
 *     并计算各个IMU时刻相对这帧LiDAR点云起始时刻的姿态，用于对点云去旋转畸变
 *  2. 从IMU预积分节点发来的高频、纯里程计姿态中读取这帧LiDAR起始时刻的位姿，用于后端LiDAR优化初值；
 *     从里程计中计算位姿增量，用于这帧LiDAR点云去平移畸变（实际没有用到平移畸变）
 *  3. 把这帧LiDAR点云投影到range image上，主要是继承了LeGO-LOAM的代码，在LIO-SAM中该操作作用不大；
 *     对这帧LiDAR点云去畸变，并发送给下一个特征提取节点提取特征
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "../include/utility.h"
#include "lio_sam/cloud_info.h"

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint16_t, ring, ring)(float, time, time))

struct OusterPointXYZIRT
{
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, t, t)(uint16_t, reflectivity, reflectivity)(uint8_t, ring, ring)(uint16_t, noise, noise)(uint32_t, range, range))

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:
    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher pubLaserCloud;

    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    //; 当前帧LiDAR点云100ms时间内，包含的IMU数据的绝对时间戳、相对于第一个点的姿态、IMU数据的个数
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];
    int imuPointerCur;     

    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;   //; ros消息收到的原始点云，转成pcl
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr fullCloud;   //; 原始点云进行去畸变之后的点云
    pcl::PointCloud<PointType>::Ptr extractedCloud;  //; 提取出的有效的点云，其实和fullCloud是一样的

    //; 能否利用IMU数据对点云去 旋转畸变，和点云消息中是否有time信息有关，只要有就可以给点云去畸变
    int deskewFlag;   
    cv::Mat rangeMat;

    //; 能否利用IMU的里程计对点云去 平移畸变，只要IMU里程计时间可以覆盖这帧LiDAR点云，就可以给点云去畸变
    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

public:
    ImageProjection() : deskewFlag(0)
    {
        //; 原始IMU测量数据
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, 
            &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        
        //; IMU预积分节点 发布的高频的、纯里程计的位姿，有全局漂移
        subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000, 
            &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        
        //; 原始lidar测量数据
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, 
            &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        //; 发布运动补偿后的点云
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/deskew/cloud_deskewed", 1);
        
        //; 发布自定义的点云消息，注意内部没有点
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1);

        allocateMemory();   
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN * Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection() {}

    void imuHandler(const sensor_msgs::Imu::ConstPtr &imuMsg)
    {
        //; 把IMU测量数据转到LIDAR坐标系下，然后存到队列中
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        // 加一个线程锁，把imu数据保存进队列
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x <<
        //       ", y: " << thisImu.linear_acceleration.y <<
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x <<
        //       ", y: " << thisImu.angular_velocity.y <<
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr &odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    //; 主要操作都在这个回调函数中，接受的消息就是激光雷达的点云消息
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        // Step 1 把点云放进队列缓存，同时判断点云是否存在ring、time等信息
        if (!cachePointCloud(laserCloudMsg))
            return;

        // Step 2 原始IMU数据准备去旋转畸变，并提供整个LIO系统初始化的位姿初值；
        // Step   IMU里程计数据准备去平移畸变，并提供后端LiDAR优化的位姿初值
        if (!deskewInfo())  
            return;

        // Step 3 把点云投影到range image上，并对点云去畸变到起始时刻进行存储
        projectPointCloud();

        // Step 4 提取出有效的点的信息并存储起来
        cloudExtraction();

        // Step 5 发布有效的点云消息，其中包括去畸变之后的点云，是发送给特征提取节点使用的
        publishClouds();

        // Step 6 一些参数的复位
        resetParameters();
    }

    /**
     * @brief 
     *   1.对点云消息放入队列进行缓存
     *   2.检查点云消息是否符合要求，比如是否有线数、时间戳等等
     * @param[in] laserCloudMsg 
     * @return true 
     * @return false 
     */
    bool  cachePointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        // cache point cloud
        // 点云数据保存进队列
        cloudQueue.push_back(*laserCloudMsg);

        //! 疑问：为什么这里要保证队列中有至少2帧点云数据？
        // 确保队列里大于两帧点云数据
        if (cloudQueue.size() <= 2)
            return false;

        // Step 1  convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn); // 转成pcl的点云格式
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // Step 2 get timestamp
        //; 一帧点云包括100ms内的所有点的数据，默认的时间戳是第一个点的时间
        cloudHeader = currentCloudMsg.header;  
        //; 这帧点云扫描的开始时间，这里可以看出来，一帧点云的时间戳是第一个点的时间
        timeScanCur = cloudHeader.stamp.toSec(); 
        //; 这帧点云扫描的结束时间，time存储的是这个点相对于起始点的时间，一般来说应该是100ms
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;  

        // Step 3 check dense flag
        // is_dense是点云是否有序排列的标志
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // Step 4 check ring channel
        // 查看驱动里是否把每个点属于哪一根扫描scan这个信息
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            //; 如果点云中有ring这个消息，那么就是包含了线数信息，就不用像loam那样计算线数了
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            // 如果没有这个信息就需要像loam或者lego loam那样手动计算scan id，现在velodyne的驱动里都会携带这些信息的
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // Step 5 check point time
        // 同样，检查是否有时间戳信息
        //; 其实这里一开始赋值为0，就是为了只检查一次是否有时间戳信息
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    /**
     * @brief 
     *   1.从IMU数据中用角速度计算各个IMU时刻的姿态，用于LiDAR点云去旋转畸变；
     *     并读取IMU姿态用于整个LIO系统初始化
     *   2.从IMU预积分发来的里程计中，计算这帧LiDAR起始时刻的位姿，用于后端LiDAR优化的初值；
     *     并计算这帧LiDAR起始和结束之间的位姿增量，用于LiDAR点云去平移畸变
     * @return true 
     * @return false 
     */
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        //! 疑问：这样不存在问题吗？IMU和LiDAR时间上是同步的，如果LiDAR发过来的时候恰恰IMU时间离它
        //!    非常近但是在他前面，那这里就不满足最后一个条件，那么返回flase，在LiDAR消息的回调函数
        //!    中就直接结束了，这帧LiDAR不就废了吗？
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            //TODO 这里改成ROS_WARN，默认log级别是Info，不会显示Debug信息，看看会不会出现这种情况？
            // ROS_DEBUG("Waiting for IMU data ...");
            ROS_WARN("Waiting for IMU data ...");
            return false;
        }

        // Step 1 计算各个IMU时刻的姿态，用于LiDAR点云去旋转畸变；并得到整个LIO系统的姿态初值
        imuDeskewInfo();

        // Step 2 计算这帧LiDAR点云的起始时刻位姿，用于后端LiDAR优化初值；并得到这帧点云的增量位姿变换
        odomDeskewInfo();

        return true;
    }


    /**
     * @brief 
     *  1.获取这帧LiDAR起始时刻的位姿，作为后端LiDAR优化的位姿初值
     *  2.计算这帧点云的100ms时间内，每个IMU数据的时刻相对于起始帧的姿态，用于LiDAR点云去畸变。
     *    这里只使用旋转进行运动补偿，因为旋转带来的点云畸变更大，而忽略平移带来的畸变
     */
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            //! 疑问：这里为什么不严格使用timeScanCur？而是前后都扩展了0.01的时间容忍？
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01) // 扔掉过早的imu
                imuQueue.pop_front();
            else
                break;
        }
        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();
            
            // Step 1 寻找这帧LiDAR点云起始时刻的IMU姿态
            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            //; 如果超过了这帧点云的最后时间戳，那么就退出循环
            if (currentImuTime > timeScanEnd + 0.01) // 这一帧遍历完了就break
                break;

            // Step 2 第一个IMU数据的姿态设置成0，后面在第一个IMU基础上用角速度积分得到各个IMU时刻的姿态
            if (imuPointerCur == 0)
            { 
                imuRotX[0] = 0;   
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;    //; 在这帧点云的100ms内，IMU数据点的索引
                continue;
            }

            // Step 3 对后面的IMU时刻，利用陀螺仪积分得到 相对第一个IMU时刻的姿态
            // get angular velocity
            double angular_x, angular_y, angular_z;
            //! 疑问：这里就是用的陀螺仪原始读数，感觉有点问题？因为没有减去零偏
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
            // 计算每一个时刻的姿态角，方便后续查找对应每个点云时间的值
            //; 角速度积分不断积分，计算欧拉角
            //! 疑问：这里为什么不用九轴IMU输出的欧拉角 - 第一个IMU时刻的欧拉角，这样不是更准吗？
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;  //; 因为最后一个循环执行后，imuPointerCur++了，所以这里要--

        if (imuPointerCur <= 0)
            return;

        // Step 4 更新标志位：可以使用IMU的姿态对LiDAR点云去畸变
        cloudInfo.imuAvailable = true;
    }


    //; 利用odom计算运动补偿的信息
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            // 扔掉过早的数据
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }
        if (odomQueue.empty())
            return;

        // 点云时间   ×××××××
        // odom时间     ×××××
        // 显然不能覆盖整个点云的时间
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        // Step 1 寻找这帧LiDAR点云起始时刻的IMU里程计位姿，作为LiDAR后端优化的位姿初值
        nav_msgs::Odometry startOdomMsg;
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }
        // 将ros消息格式中的姿态转成tf的格式
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);
        // 然后将四元数转成欧拉角
        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw = yaw;

        //; 置位标志位：odom提供了这一帧点云的初始位姿
        cloudInfo.odomAvailable = true;


        // get end odometry at the end of the scan
        // Step 2 寻找覆盖这帧LiDAR点云的最后一个点的位姿，从而得到这帧LiDAR点云的增量位姿，用于平移去畸变
        odomDeskewFlag = false;
        // 这里发现没有覆盖到最后的点云，那就不能用odom数据来做运动补偿
        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;
        // 找到点云最晚时间对应的odom数据
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }
        // 这个代表odom退化了，就置信度不高了，就没有必要使用odom来进行运动补偿了
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;
        // 起始位姿和结束位姿都转成Affine3f这个数据结构
        Eigen::Affine3f transBegin = pcl::getTransformation(
            startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(
            endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 计算起始位姿和结束位姿之间的delta pose
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
        // 将这个增量转成xyz和欧拉角的形式
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true; // 表示可以用odom来做运动补偿
    }

    /**
     * @brief 给定点云的绝对时间戳，寻找此时它相对于这帧点云的起始时刻的相对姿态
     */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0;
        *rotYCur = 0;
        *rotZCur = 0;

        int imuPointerFront = 0;
        // imuPointerCur是imu计算的旋转buffer的总共大小，这里用的就是一种朴素的确保不越界的方法
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        // imuPointerBack     imuPointerFront
        //       ×                   ×
        //                 ×
        //           imuPointerCur
        // 如果时间戳不在两个imu的旋转之间，就直接赋值了
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        }
        else
        {
            // 否则 做一个线性插值，得到相对旋转
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0;
        *posYCur = 0;
        *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.
        
        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;
        // float ratio = relTime / (timeScanEnd - timeScanCur);
        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    /**
     * @brief 对LiDAR点云进行去畸变，对齐到一帧LiDAR点云的起始时刻坐标系下。
     *  注意这里只对旋转进行了去畸变，没有使用平移去畸变
     * 
     * @param[in] point    原始的LiDAR点
     * @param[in] relTime  这个点到这帧LiDAR点云的相对时间，0-100ms范围内
     * @return PointType 
     */
    PointType deskewPoint(PointType *point, double relTime)
    {
        //; deskewFlag 其实是反应点云消息中是否有时间戳信息，如果有才能用来运动补偿。有的话deskewFlag=1,否则-1
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;
        
        // Step 1 计算当前点相对于这帧点云起始时刻的位姿
        // relTime是相对时间，加上起始时间就是绝对时间
        double pointTime = timeScanCur + relTime;
        float rotXCur, rotYCur, rotZCur;
        // 计算当前点相对起始点的相对旋转
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);
        // 这里没有计算平移补偿
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        // Step 2 计算第一个点相对起始时刻的位姿，因为之前计算的其实时刻并不是第一个点，而是有0.01s的时间容忍
        if (firstPointFlag == true)
        {
            // 计算第一个点的相对位姿
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // Step 3 计算当前点 相对 真正的这帧点云第一个点时刻的相对位姿
        // 计算当前点和第一个点的相对位姿
        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        // Step 4 把当前点的位置补偿到这帧点云的第一个点时刻的坐标系下，得到去畸变之后的点
        PointType newPoint;
        // 就是R × p + t，把点补偿到第一个点对应时刻的位姿
        newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y + transBt(0, 2) * point->z + transBt(0, 3);
        newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y + transBt(1, 2) * point->z + transBt(1, 3);
        newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y + transBt(2, 2) * point->z + transBt(2, 3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }
    
    
    /**
     * @brief 
     *  1. LiDAR点云投影到range image上，为下一步处理做准备
     *  2. 对LiDAR点云去畸变，转到这帧LiDAR点云起始时刻的坐标系下
     */
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            // 取出对应的某个点
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            // 计算这个点距离lidar中心的距离，其实就是点到坐标原点的距离
            float range = pointDistance(thisPoint);
            // 距离太小或者太远都认为是异常点
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            // Step 1 计算LiDAR点在range image中的行坐标
            // 取出对应的在第几根scan上
            int rowIdn = laserCloudIn->points[i].ring;
            // scan id必须合理
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 如果需要降采样，就根据scan id适当跳过
            if (rowIdn % downsampleRate != 0)
                continue;
            // 计算水平角，-180 ~ +180之间, 注意这里算的是atan2（x/y）
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            // Step 2 计算LiDAR点在range image中的列坐标
            static float ang_res_x = 360.0 / float(Horizon_SCAN);  //; Horizon_SCAN = 1800
            //; 计算水平线束id，转换到x负方向为起始0度，逆时针为正方向，从0变到H
            //;                                ^ +x, angle=90°, id=900
            //;                                |
            //;                                |         
            //;  +y, angle=0°, id=1350 <----------------  -y, angle=+180°, id=450
            //;                                | 
            //;                                |
            //;       -x, angle=-90°, id=1800    -x, angle=-90°, id=0
            // 注意round是四舍五入函数, round(1.6) = 2， round(-1.6) = -2。
            // x+上的点，坐标(1,0)，ang=90, id = 900;
            // y+上的点，坐标(0,1), ang=0,  id = 90/0.2 + 900 = 1350;
            // x-上的点，坐标(-1,0),ang=-90, id = 180/0.2 + 900 = 1800;
            // y-上的点向x上，坐标(0,-1),ang=+180, id = -90/0.2 + 900 = 900 - 450 = 450
            // y-上的点向x下，坐标(0,-1),ang=-180, id = 270/0.2 + 900 = 2250 -> 2250 - 1800 = 450
            int columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;
            // 对水平id进行检查
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            // 如果这个位置已经有填充了就跳过
            //; 行id是线数，列id是水平方向的一周的id，范围0-1800
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            // 将这个点的距离数据保存进这个range矩阵中
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            
            // Step 3 对这个LiDAR点取旋转畸变，然后存到去畸变后的点云中
            // 算出这个点的索引
            int index = columnIdn + rowIdn * Horizon_SCAN;
            // 对点做运动补偿，使用的是IMU算出来的旋转进行运动补偿
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            // 保存这个点的坐标
            fullCloud->points[index] = thisPoint;
        }
    }
    
    /**
     * @brief 提取有效的点，存储到cloudInfo和extractedCloud中
     * 
     */
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        // 遍历每一根scan
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 这个scan可以计算曲率的起始点（计算曲率需要左右各五个点）
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i, j) != FLT_MAX)
                {
                    // 这是一个有用的点
                    // mark the points' column index for marking occlusion later
                    // 这个点对应着哪一根垂直线，也就是水平方向上1800哪个索引
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    // 他的距离信息
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
                    // save extracted cloud
                    // 他的3d坐标信息
                    extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                    // size of extracted cloud
                    // count只在有效点才会累加
                    ++count;
                }
            }
            // 这个scan可以计算曲率的终点
            cloudInfo.endRingIndex[i] = count - 1 - 5;
        }
    }

    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        // Step 1 把去畸变之后的点云发布，并存到自定义的cloudInfo中
        //; 注意这个函数还挺有意思的，里面主要功能是把pcl点云转成ros消息并通过函数返回值返回，但是里面也发布了消息
        cloudInfo.cloud_deskewed = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        
        //; 这里发布的是自定义的点云消息类型
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;

    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();

    return 0;
}
