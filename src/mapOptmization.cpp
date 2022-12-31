/**
 * @file mapOptmization.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-31
 *  1. 执行后端scan-to-map优化
 *  2. 如果有回环/gps信息，则加入因子图中进行位姿图优化
 *  3. 其他线程：回环检测线程、rviz显示线程
 * @copyright Copyright (c) 2022
 * 
 */
#include "../include/utility.h"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

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

using namespace gtsam;

using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

// A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

class mapOptimization : public ParamServer
{
public:
    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    lio_sam::cloud_info cloudInfo;

    //; 关键帧对应的点云，最后保存全局地图的时候保存的就是这些点云。如果当前帧被判断为关键帧了，那么就会把这一帧的点云存进去
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    //; 以下两个变量是用点云数据结构来存储关键帧的 3D位置 或者 6D姿态，注意它是轨迹而不是点云，这里只是借用了点云的结构
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;     //  存储关键帧的位置信息的点云
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // 存储关键帧的6D位姿信息的点云
    //; 这两个变量就是上面两个变量的拷贝，主要是在回环检测中使用，为了防止回环锁住线程锁导致后端的主线程需要一直等待
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    //; 下面的就是当前帧收到的点云，这里写的是Last，实际就是最新的点云。后面的两个DS是downsample，即降采样之后的点云
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;   // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;     // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;   // downsampled surf featuer set from odoOptimization

    //; 下面两个存储当前帧的点云匹配到局部地图中求解位姿的时候，laserCloudOri是当前帧点云的点，coeffSel.xyz是这个点到
    //; 局部地图的平面的法向量，coeffSel.intensity是这个点到局部地图的平面的距离，也就是残差
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    //; 注意这个变量很重要，这个就是存储了和当前帧进行匹配的局部地图，结构如下：
    //; map< 关键帧的索引， < 面点地图， 角点地图> >
    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer; // 局部地图的一个容器
    
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS; // 角点局部地图的下采样后的点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;   // 面点局部地图的下采样后的点云

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap; // 角点局部地图的kdtree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;   // 面点局部地图的kdtree

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum = 0; // 当前局部地图下采样后的角点数目
    int laserCloudSurfFromMapDSNum = 0;   // 当前局部地图下采样后的面点数目
    int laserCloudCornerLastDSNum = 0;    // 当前帧下采样后的角点的数目
    int laserCloudSurfLastDSNum = 0;      // 当前帧下采样后的面点数目

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    mapOptimization()
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal = nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 1);
        pubPath = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

        //; 订阅上一个特征提取节点发来的点云信息，主要是线点和面点信息
        subCloud = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1,
                    &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        
        //; 订阅的回环检测的消息，注意这个是手动指定哪两帧是回环的，作者最后也没用它
        subLoop = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());
        
        // 订阅一个保存地图功能的服务
        srvSaveMap = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        // 体素滤波设置珊格大小
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }

    // 预先分配内存
    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());   // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());     // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());   // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i)
        {
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }


    /**
     * @brief 后端优化的主处理函数
     */
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr &msgIn)
    {
        // extract time stamp
        // 提取当前时间戳
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info and feature cloud
        // Step 0 提取cloudinfo中的角点和面点，并转成pcl格式存储
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        // 控制后端频率，两帧处理一帧
        //; mappingProcessInterval在配置文件中是0.15秒，而点云消息的频率是0.1秒，
        //; 如果这里算力足够的话，可以设置每一帧都进行处理
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            // Step 1 更新当前帧scan-to-map的lidar位姿初值
            updateInitialGuess();

            // Step 2 提取当前帧相关的关键帧并且构建点云局部地图，最后对点云局部地图也进行了下采样
            extractSurroundingKeyFrames();

            // Step 3 对当前帧进行下采样，也是为了降低后面点云配准的计算量
            downsampleCurrentScan();

            // Step 4 对scan-to-map的点云配准进行优化求解
            scan2MapOptimization(); 
          
            // Step 5 根据优化后的当前帧位姿，确定的当前帧是否是关键帧
            saveKeyFramesAndFactor();

            // Step 6 从因子图中取出优化后的结果，并更新到全局变量中，调整全局轨迹
            correctPoses();

            // Step 7 发布最新的lidar里程计，包括里程计消息、tf变换、给前端IMU预积分矫正零偏使用的lidar增量里程计
            publishOdometry();

            // Step 8 发送各种rviz可视化信息
            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr &gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const *const pi, PointType *const po)
    {
        po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
        po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
        po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
// 使用openmp进行并行加速
#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            // 每个点都施加RX+t这样一个过程，所以说就可以使用openmp进行加速
            cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
            cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
            cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                            gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
    }
    // 转成gtsam的数据结构
    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                            gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw = transformIn[2];
        return thisPose6D;
    }

    /**
     * @brief 保存地图的service的处理函数
     */
    bool saveMapService(lio_sam::save_mapRequest &req, lio_sam::save_mapResponse &res)
    {
        string saveMapDirectory;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        // 如果是空，说明是程序结束后的自动保存，否则是中途调用ros的service发布的保存指令
        if (req.destination.empty())
            saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
        else
            saveMapDirectory = std::getenv("HOME") + req.destination;
        cout << "Save destination: " << saveMapDirectory << endl;
        // create directory and remove old files;
        // 删掉之前有可能保存过的地图
        int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
        // save key frame transformations
        // 首先保存关键帧轨迹
        pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
        // extract global point cloud map
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        
        // 遍历所有关键帧，将点云全部转移到世界坐标系下去
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
        {
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            // 类似进度条的功能
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        // 如果没有指定分辨率，就是直接保存
        if (req.resolution != 0)
        {
            cout << "\n\nSave resolution: " << req.resolution << endl;

            // down-sample and save corner cloud
            // 使用指定分辨率降采样，分别保存角点地图和面点地图
            downSizeFilterCorner.setInputCloud(globalCornerCloud);
            downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
            downSizeFilterCorner.filter(*globalCornerCloudDS);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
            // down-sample and save surf cloud
            downSizeFilterSurf.setInputCloud(globalSurfCloud);
            downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
            downSizeFilterSurf.filter(*globalSurfCloudDS);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
        }
        else
        {
            // save corner cloud
            pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
            // save surf cloud
            pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
        }

        // save global point cloud map
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        // 保存全局地图
        int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
        res.success = ret == 0;

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed\n"
             << endl;

        return true;
    }

    // 全局可视化线程
    void visualizeGlobalMapThread()
    {
        // 更新频率设置为0.2hz
        ros::Rate rate(0.2);
        while (ros::ok())
        {
            rate.sleep();
            publishGlobalMap();
        }
        // 当ros被杀死之后，执行保存地图功能
        if (savePCD == false)
            return;

        lio_sam::save_mapRequest req;
        lio_sam::save_mapResponse res;

        if (!saveMapService(req, res))
        {
            cout << "Fail to save map" << endl;
        }
    }

    // 发布可视化全局地图
    void publishGlobalMap()
    {
        // 如果没有订阅者就不发布，节省系统负载
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;
        // 没有关键帧自然也没有全局地图了
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        // 把所有关键帧送入kdtree
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        // 寻找具体最新关键帧一定范围内的其他关键帧
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();
        // 把这些找到的关键帧的位姿保存起来
        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        // 最一个简单的下采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;                                                                                            // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for (auto &pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            // 找到这些下采样后的关键帧的索引，并保存下来
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
        {
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            // 将每一帧的点云通过位姿转到世界坐标系下
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        // 转换后的点云也进行一个下采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        // 最终发布出去
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }

    // 回环检测线程
    void loopClosureThread()
    {
        // 如果不需要进行回环检测，那么就退出这个线程
        if (loopClosureEnableFlag == false)
            return;
        // 设置回环检测的频率，配置参数中设置的是1hz
        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            // 执行完一次就必须sleep一段时间，否则该线程的cpu占用会非常高
            rate.sleep();
            // 执行回环检测
            performLoopClosure();
            // 检测到的回环信息进行可视化
            visualizeLoopClosure();
        }
    }

    // 接收外部告知的回环信息
    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr &loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        // 回环信息必须是配对的，因此大小检查一下是不是2
        if (loopMsg->data.size() != 2)
            return;
        // 把当前回环信息送进队列
        loopInfoVec.push_back(*loopMsg);
        // 如果队列里回环信息太多没有处理，就把老的回环信息扔掉
        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performLoopClosure()
    {
        // 如果没有关键帧，就没法进行回环检测了
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        // 把存储关键帧的位姿的点云copy出来，避免线程冲突
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;  //; 3D是关键帧的xyz, 不带rpy
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;  //; 6D是关键帧的xyz, rpy
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        // 首先看一下外部通知的回环信息
        //; 外部通知的回环消息就是可以手动设置两帧之间存在回环关系
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            // 然后根据里程记的距离来检测回环
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;
        // 检出回环之后开始计算两帧位姿变换
        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            // 稍晚的帧（当前帧）就把自己取了出来，因为后面的参数是0
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            // 稍早一点（回环帧）的就把自己和周围一些点云取出来，也就是构成一个帧到局部地图的一个匹配问题
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            // 如果点云数目太少就算了
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                // 把局部地图发布出去供rviz可视化使用
                publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        // 使用简单的icp来进行帧到局部地图的配准
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        // 设置两个点云
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        //; 这个unused_result是因为下面的点云配准函数必须要求一个变量传入，但是这个变量我们用不到
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        // 执行点云配准
        icp.align(*unused_result);
        // 检查icp是否收敛且得分是否满足要求
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        // 把修正后的当前点云发布供可视化使用
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        // 获得两个点云的变换矩阵结果
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        // 找到稍晚帧的位姿结果
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        // 将icp的结果补偿过去，就是稍晚帧的更为准确的位姿结果
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
        // 将稍晚帧的位姿转成平移+欧拉角
        pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);

        //! 感觉这里有点找麻烦？不就是为了得到稍早帧和稍帧之间的坐标变换关系吗？但是你上面点云配准不是得到了吗？
        // from是修正后的稍晚帧的点云位姿
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // to是修正前的稍早帧的点云位姿
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        // 使用icp的得分作为他们的约束的噪声项
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        // 将两帧索引，两帧相对位姿和噪声作为回环约束送入队列
        //; 这里是把回环检测的结果转成gtsam的格式
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        // 保存已存在的约束对
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    //; 这个回环检测的方式很简单，就是在kdtree中寻找和当前帧半径15m内的关键帧，并且这个关键帧离当前帧
    //; 时间超过30秒，那么就认为存在回环。当然这里是存在一些问题的，比如当前在等红绿灯，那么时间肯定超过了30秒
    //; 此时就会把当前帧前面的相邻帧认为是回环帧，这是存在问题的。此时可以通过增加判断点云的序列号来进一步筛选回环帧
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        // 检测最新帧是否和其他帧形成回环，所以后面一帧的id就是最后一个关键帧
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        // 检查一下较晚帧是否和别的形成了回环，如果有就算了
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        // 把只包含关键帧位移信息的点云填充kdtree
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        // 根据最后一个关键帧的平移信息，寻找离他一定距离内的其他关键帧，配置参数中是在15m半径内进行搜索
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        // 遍历找到的候选关键帧
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            // 必须满足时间上超过一定阈值，才认为是一个有效的回环，配置文件中是30秒
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                // 此时就退出了
                //; 注意这里只要找到一个就直接退出
                loopKeyPre = id;
                break;
            }
        }
        // 如果没有找到回环或者回环找到自己身上去了，就认为是此次回环寻找失败
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        // 作者表示这个功能还没有使用过，可以忽略它
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        // 外部的先验回环消息队列
        if (loopInfoVec.empty())
            return false;
        // 取出回环信息，这里是把时间戳作为回环信息
        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();
        // 如果两个回环帧之间的时间差小于30s就算了
        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;
        // 如果现存的关键帧小于2帧那也没有意义
        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        // 遍历所有的关键帧，找到离后面一个时间戳更近的关键帧的id
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        // 同理，找到距离前一个时间戳更近的关键帧的id
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }
        // 两个是同一个关键帧，就没什么意思了吧
        if (loopKeyCur == loopKeyPre)
            return false;
        // 检查一下较晚的这个帧有没有被其他时候检测了回环约束，如果已经和别的帧形成了回环就算了
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;
        // 两个帧的索引输出
        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        // searchNum是搜索范围
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            // 找到这个idx
            int keyNear = key + i;
            // 如果超出范围就算了
            if (keyNear < 0 || keyNear >= cloudSize)
                continue;
            // 否则把对应角点和面点的点云转到世界坐标系下去
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        }
        // 如果没有有效的点云就算了
        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        // 把点云下采样
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    // 将已有的回环约束可视化出来
    void visualizeLoopClosure()
    {
        // 如果没有回环约束就算了
        if (loopIndexContainer.empty())
            return;

        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        // 回环约束的两帧作为node添加
        // 先定义一下node的信息
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3;
        markerNode.scale.y = 0.3;
        markerNode.scale.z = 0.3;
        markerNode.color.r = 0;
        markerNode.color.g = 0.8;
        markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        // 两帧之间约束作为edge添加
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9;
        markerEdge.color.g = 0.9;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;
        // 遍历所有回环约束
        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            // 添加node和edge
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        // 最后发布出去供可视化使用
        pubLoopConstraintEdge.publish(markerArray);
    }

    
    /**
     * @brief 更新本次LiDAR优化的位姿初值
     *  1. 如果是整个系统的第一帧，则使用IMU的姿态赋值整个LIO系统的初始姿态
     *  2. 如果IMU预积分的高频增量里程计位姿可用，则计算本次收到的里程计和上次收到的里程计之间的增量位姿，
     *     然后叠加到上次优化得到的最优位姿上，得到本次优化的位姿的估计值
     *  3. 如果IMU预积分的高频增量里程计位姿不可用，但是9轴IMU的姿态可用，则只使用其姿态，和上面的方法一样
     *     计算本次优化的姿态（不包括平移）的估计值
     */
    void updateInitialGuess()
    {
        // save current transformation before any processing
        //; transformTobeMapped是上一帧优化后的最佳位姿，是包含回环的全局位姿，无漂移
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped); //; 调用pcl接口转成eigen数据结构

        //; 上次9轴IMU给出的姿态，注意平移置为0了，因为平移不准
        static Eigen::Affine3f lastImuTransformation;

        // initialization
        // Step 1 系统还没有初始化，也就是当前是第一帧LiDAR点云，则直接赋值初始位姿即可
        //; 注意Poses3D只是位置，而Poses6D是位置+姿态
        if (cloudKeyPoses3D->points.empty())
        {
            // 初始的位姿就由磁力计提供，因为这里使用的是9轴的imu
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            // 无论vio还是lio 系统的不可观都是4自由度，平移+yaw角，这里虽然有磁力计将yaw对齐，但是也可以考虑不使用yaw
            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            // 保存磁力计得到的位姿，平移置0
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;

            //! 感觉这里有点bug，应该把lastImuPreTransformation也赋值成磁力计位姿的，
            //! 这样第1帧来的时候（现在是第0帧），就直接可以处理了
        }

        // use imu pre-integration estimation for pose guess
        // Step 2 如果有IMU预积分节点发来的高频的增量里程计位姿，则计算里程计增量，叠加到上一次优化的
        // Step   全局位姿上，从而得到这次的全局位姿的估计值
        static bool lastImuPreTransAvailable = false;     //; 上一帧是否有IMU预积分的里程计位姿
        static Eigen::Affine3f lastImuPreTransformation;  //; 上一帧发来的IMU预积分里程计位姿
        if (cloudInfo.odomAvailable == true)
        {
            // 将提供的初值转成eigen的数据结构
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            //; 如果是第一次收到，则备份下来
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;  // 将当前里程记结果记录下来
                lastImuPreTransAvailable = true;  // 收到第一个里程记数据以后这个标志位就是true
            }
            //; 否则不是第一次收到，就计算里程计位姿的增量，再叠加到上次的全局优化结果上
            else
            {
                // 计算上一个里程记的结果和当前里程记结果之间的delta pose
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                // 将这个增量加到上一帧最佳位姿上去，就是当前帧位姿的一个先验估计位姿
                Eigen::Affine3f transFinal = transTobe * transIncre;
                // 将eigen变量转成欧拉角和平移的形式
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                  transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                // 同理，把当前帧的值保存下来
                lastImuPreTransformation = transBack;
                // 虽然有里程记信息，仍然需要把imu磁力计得到的旋转记录下来
                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                
                //; 注意这里直接return了，也就是我们是优先使用IMU里程计积分的结果来做预测
                return;   
            }  
        }

        // use imu incremental estimation for pose guess (only rotation)
        // Step 3 如果没有IMU预积分的高频里程计位姿，则使用9轴IMU的姿态作为位姿初值
        // 如果没有里程记信息，就是用imu的旋转信息来更新，因为单纯使用imu无法得到靠谱的平移信息，因此平移直接置0
        if (cloudInfo.imuAvailable == true)
        {
            // 初值计算方式和上面相同，只不过注意平移置0
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses - 1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }
    
    /**
     * @brief 提取最后一个关键帧周围的一些关键帧，构成局部地图
     */
    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;     // 保存kdtree提取出来的元素的索引
        std::vector<float> pointSearchSqDis; // 保存距离查询位置的距离的数组

        // extract all the nearby key poses and downsample them
        // Step 1 根据关键帧位置，构造kdtree，方便后面差最后啊
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        
        // Step 2 根据最后一个KF的位置，提取一定距离内的关键帧
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        // 根据查询的结果，把这些点的位置存进一个点云结构中
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        // Step 3 避免关键帧过多，因此做一个下采样
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        
        // Step 4 从真正存在的关键帧中，寻找距离下采样后的关键帧最近的那个关键帧，最后组成真正的局部关键帧
        // 确认每个下采样后的点的索引，就使用一个最近邻搜索，其索引赋值给这个点的intensity数据位
        for (auto &pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        // Step 5 刚刚是提取了一些空间上比较近的关键帧，然后再提取一些时间上比较近的关键帧
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses - 1; i >= 0; --i)
        {
            // 最近十秒的关键帧也保存下来
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        // Step 6 根据筛选出来的关键帧和对应的特征点，构成用于本次匹配的局部地图
        extractCloud(surroundingKeyPosesDS);
    }

    /**
     * @brief 输入关键帧，把这些关键帧对应的点云组合起来构成局部地图
     */
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        // Step 1 先清空本次匹配要使用的局部地图，因为要重新构造，上一帧的局部地图这次不能用了
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();

        // Step 2 构造本次要使用的局部地图
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 简单校验一下关键帧距离不能太远，这个实际上不太会触发
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;
            // 取出提出出来的关键帧的索引
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;

            // Step 2.1 如果这个关键帧对应的全局点云已经转换过，则直接拿过来即可
            //; 这里就是为了降低运算量，比如当前帧的局部地图有一部分是和上一帧的局部地图重合的，
            //; 这里就没有必要重复转换到世界坐标系下了，直接使用之前转化后存储的结果即可
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())
            {
                // transformed cloud available
                // 直接从容器中取出来加到局部地图中
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
            }
            // Step 2.2 否则这个关键帧的点云没有转换过，那就通过该帧的位姿把该帧点云转到世界坐标系下
            else
            {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                // 点云转换之后加到局部地图中
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap += laserCloudSurfTemp;
                //; 把转换后的面点和角点存进这个容器中，方便后续直接加入点云地图，避免点云转换的操作，节约时间
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
        }

        // Step 3 对局部地图进行降采样，降低计算量
        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        // Step 4 如果这个局部地图容器过大，就clear一下，避免占用内存过大 
        //; 这里直接clear掉，下次再进入上面的for循环就相当于直接重新来过，看起来好像有点暴力，
        //; 但是实际上来说1000帧之前的点云很难对现在的点云还有影响，所以这里clear掉也是可以的
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    /**
     * @brief 按照距离，选择当前帧周围的关键帧，构成局部地图
     * 
     */
    void extractSurroundingKeyFrames()
    {
        // 如果当前没有关键帧，就return了
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        extractNearby();
    }

    /**
     * @brief 当前帧的角点和面点分别进行下采样，为了减少计算量
     */
    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        // 将欧拉角转成eigen的对象
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization()
    {
        updatePointAssociateToMap();  //; 把当前帧的先验位姿转成eigen对象
// 使用openmp并行加速，也就是加速下面的for循环
//; 使用的条件是下面的每一次循环都是相互独立的
#pragma omp parallel for num_threads(numberOfCores)
        // 遍历当前帧的角点
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            // 将该点从当前帧通过初始的位姿转换到地图坐标系下去
            pointAssociateToMap(&pointOri, &pointSel);
            // 在角点地图里寻找距离当前点比较近的5个点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
            // 计算找到的点中距离当前点最远的点，如果距离太大那说明这个约束不可信，就跳过
            //; pcl搜索返回的距离是从小到大排序的
            if (pointSearchSqDis[4] < 1.0)
            {
                float cx = 0, cy = 0, cz = 0;
                // 计算协方差矩阵
                // 首先计算均值
                for (int j = 0; j < 5; j++)
                {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5;  //; 计算均值
                cy /= 5;
                cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++)
                {
                    //; 计算协方差，减去均值
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax;  //; 协方差矩阵是对称的，只需要计算对角线上面的
                    a12 += ax * ay;
                    a13 += ax * az;
                    a22 += ay * ay;
                    a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;
                //; matA1就是协方差矩阵
                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;
                // 特征值分解，matD1中存储的就是特征值
                cv::eigen(matA1, matD1, matV1);
                // 这是线特征性，要求最大特征值大于3倍的次大特征值
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
                {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 特征向量对应的就是直线的方向向量，matV1(0,123)就是最大特征值对应的特征向量的方向
                    // 通过点的均值往两边拓展，构成一个线的两个端点
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);
                    // 下面是计算点到线的残差和垂线方向（及雅克比方向）
                    float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));
                    float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
                    float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;
                    float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;
                    float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;
                    float ld2 = a012 / l12;

                    // 一个简单的核函数，残差越大权重降低
                    //; 点离直线的距离越近，认为越可靠，因此距离越远（残差越大）权重越低
                    float s = 1 - 0.9 * fabs(ld2);

                    //; 把鲁棒核函数的乘上垂线方向和残差
                    //! 这里垂线方向就是梯度下降的方向
                    //! 不太理解，梯度下降的方向不应该是计算正则方程的时候得到的吗？
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;
                    // 如果残差小于10cm，就认为是一个有效的约束
                    if (s > 0.1)
                    {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;  //; 标志位置true，说明残差有效
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            // 同样找5个面点
            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;
            // 平面方程Ax + By + Cz + 1 = 0
            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();
            // 同样最大距离不能超过1m
            if (pointSearchSqDis[4] < 1.0)
            {
                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                // 求解Ax = b这个超定方程
                matX0 = matA0.colPivHouseholderQr().solve(matB0);
                // 求出来x的就是这个平面的法向量
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                // 归一化，将法向量模长统一为1
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    // 每个点代入平面方程，计算点到平面的距离，如果距离大于0.2m认为这个平面曲率偏大，就是无效的平面
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }
                // 如果通过了平面的校验
                if (planeValid)
                {
                    // 计算当前点到平面的距离
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                    // 分母不是很明白，为了更多的面点用起来？
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
                    
                    //; 平面的法向量，就是梯度下降的方向？
                    //! 梯度下降的方向和这个法向量有什么关系？
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1)
                    {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;  //; 梯度下降的法向量
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    // 将角点约束和面点约束统一到一起
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
        {
            // 只有标志位为true的时候才是有效约束
            if (laserCloudOriCornerFlag[i] == true)
            {
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
        {
            if (laserCloudOriSurfFlag[i] == true)
            {
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        // 标志位清零
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    //; 这里实际上使用的高斯-牛顿的优化算法，并不是LM算法
    bool LMOptimization(int iterCount)
    {
        // 原始的loam代码是将lidar坐标系转到相机坐标系，这里把原先loam中的代码拷贝了过来，
        // 但是为了坐标系的统一，就先转到相机系优化，然后结果转回lidar系
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        // 将lidar系的先验位姿转到相机系
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        //; 要有足够的约束才能进行优化
        if (laserCloudSelNum < 50)
        {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++)
        {
            // 首先将当前点以及点到线（面）的单位向量转到相机系
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            // 相机系下的旋转顺序是Y - X - Z对应lidar系下Z -Y -X
            float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y + (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;
            float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x + ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;
            float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y + ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;
            // lidar -> camera
            // 这里就是把camera转到lidar了
            //! 问题：点到直线的距离和点到平面的距离定义为残差，那么这个残差关于点的雅克比就是
            //! 点到直线（平面）的方向吗？该怎么计算呢？
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            //; 关于平移t的雅克比 
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;  //; J^T*J*x = -J^T*e，所以这里直接把符号加到等号右边的J^T上了
        }  
        // 构造JTJ以及-JTe矩阵
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        // 求解增量
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0)
        {
            // 检查一下是否有退化的情况
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
            // 对JTJ进行特征值分解
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--)
            {
                // 特征值从小到大遍历，如果小于阈值就认为退化
                //; 特征值如果比较小，说明解存在一个零空间，存在退换现象。比如在空旷的走廊里，走廊方向的平移是无法估计的
                //! 没有很明白这个退化和零空间的问题
                if (matE.at<float>(0, i) < eignThre[i])
                {
                    // 对应的特征向量全部置0
                    for (int j = 0; j < 6; j++)
                    {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;  //; 退化标志位置位
                }
                else
                {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }
        // 如果发生退化，就对增量进行修改，退化方向不更新
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }
        // 增量更新
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);
        // 计算更新的旋转和平移大小
        float deltaR = sqrt(
            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
            pow(matX.at<float>(3, 0) * 100, 2) +
            pow(matX.at<float>(4, 0) * 100, 2) +
            pow(matX.at<float>(5, 0) * 100, 2));
        // 旋转和平移增量足够小，认为优化问题收敛了
        //; 0.05度和0.05厘米
        if (deltaR < 0.05 && deltaT < 0.05)
        {
            return true; // converged
        }
        // 否则继续优化
        return false; // keep optimizing
    }

    /**
     * @brief 
     * 
     */
    void  scan2MapOptimization()
    {
        // 如果没有关键帧，那也没办法做当前帧到局部地图的匹配
        if (cloudKeyPoses3D->points.empty())
            return;

        // 判断当前帧的角点数和面点数是否足够: 10个角点，100个面点
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // 分别把角点面点局部地图构建kdtree
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
            // Step 1 迭代求解
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization(); //; 边缘点的残差计算
                surfOptimization();   //; 面点的残差计算

                combineOptimizationCoeffs(); //; 结合两个部分的雅克比

                if (LMOptimization(iterCount) == true) //; 执行优化
                    break;
            }

            // Step 2 把scan-to-map优化后的姿态和9轴IMU的姿态进行一定比例的融合
            transformUpdate();
        }
        else
        {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    /**
     * @brief 把scan-to-map优化后的姿态和9轴IMU的姿态进行一定比例的融合
     */
    void transformUpdate()
    {
        // Step 1 如果IMU的姿态可用，则对roll和pitch角进行一定比例的融合
        if (cloudInfo.imuAvailable == true)
        {
            // 因为roll 和 pitch原则上全程可观，因此这里把lidar推算出来的姿态和磁力计结果做一个加权平均
            // 首先判断车翻了没有，车翻了好像做slam也没有什么意义了，当然手持设备可以pitch很大，这里主要避免插值产生的奇异
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                //; imu插值的权重，在配置文件中，0.01，说明更加相信lidar里程计的结果
                double imuWeight = imuRPYWeight; 
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                // lidar匹配获得的roll角转成四元数
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                // imu获得的roll角
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                // 使用四元数球面插值
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                // 插值结果作为roll的最终结果
                transformTobeMapped[0] = rollMid;

                // 下面pitch角同理
                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        // Step 2 对roll pitch和z进行一些约束，主要针对室内2d场景下，已知2d先验可以加上这些约束
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
        
        //; incrementalOdometryAffineBack是本次scan-to-map优化后的位姿，暂时还没有包括回环
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    // 当前帧是否为关键帧
    bool saveFrame()
    {
        // 如果没有关键帧，那直接认为是关键帧
        if (cloudKeyPoses3D->points.empty())
            return true;
        // 取出上一个关键帧的位姿
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 当前帧的位姿转成eigen形式
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 计算两个位姿之间的delta pose
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        // 转成平移+欧拉角的形式
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);
        // 任何一个旋转大于给定阈值或者平移大于给定阈值就认为是关键帧
        if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        // 如果是第一帧关键帧
        if (cloudKeyPoses3D->points.empty())
        {
            // 置信度就设置差一点，尤其是不可观的平移和yaw角
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            // 增加先验约束，对第0个节点增加约束
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // 加入节点信息
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }
        else
        {
            // 如果不是第一帧，就增加帧间约束
            // 这时帧间约束置信度就设置高一些
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            // 转成gtsam的格式
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
            // 这是一个帧间约束，分别输入两个节点的id，帧间约束大小以及置信度
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            // 加入节点信息
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        // 如果没有gps信息就算了
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        // 如果有gps但是没有关键帧信息也算了
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            // 第一个关键帧和最后一个关键帧相差很近也算了，要么刚起步，要么会触发回环
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        // gtsam反馈的当前x，y的置信度，如果置信度比较高也不需要gps来打扰
        if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4, 4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            // 把距离当前帧比较早的帧都抛弃
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            // 比较晚就索性再等等lidar计算
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                // 说明这个gps时间上距离当前帧已经比较近了,那就把这个数据取出来
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                // 如果gps的置信度不高，也没有必要使用了
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;
                // 取出gps的位置
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                // 通常gps的z没有x y准，因此这里可以不使用z值
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                // gps的x或者y太小说明还没有初始化好
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                // 加入gps观测不宜太频繁，相邻不超过5m
                //! 这里xq视频中讲的不太对，论文中说的是只要是前面比较近的地方加了GPS，那么当前帧的位置也不会飘的太远，
                //! 所以没有必要非常频繁的加入GPS。但是我觉得频繁加入问题也不大吧？
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                // gps的置信度，标准差设置成最小1m，也就是不会特别信任gps信号
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                // 调用gtsam中集成的gps的约束
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);
                // 加入gps之后等同于回环，需要触发较多的isam update
                aLoopIsClosed = true;
                break;
            }
        }
    }

    void addLoopFactor()
    {
        // 有一个专门的回环检测线程会检测回环，检测到就会给这个队列塞入回环结果
        if (loopIndexQueue.empty())
            return;
        // 把队列里所有的回环约束都添加进来
        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            // 这是一个帧间约束
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            // 回环的置信度就是icp的得分
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            // 加入约束
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }
        // 清空回环相关队列
        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        // 标志位置true
        aLoopIsClosed = true;
    }

    /**
     * @brief 
     *  1. 根据scan-to-map优化后的位姿判断当前帧是否是关键帧
     *  2. 如果有GPS或回环等全局观测信息，则执行位姿图优化
     */
    void saveKeyFramesAndFactor()
    {
        // Step 1 通过旋转和平移的增量来判断是否是关键帧
        if (saveFrame() == false)
            return;

        // Step 2 如果是关键帧，则添加到因子图中，执行位姿图优化
        // Step 2.1 添加因子 和 优化变量的初值
        // odom factor
        addOdomFactor();
        // gps factor
        addGPSFactor();
        // loop factor
        addLoopFactor();

        // Step 2.2 把因子和优化变量初值更新到因子图中，并执行一次优化
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        // Step 2.3 如果加入了gps的约束或者回环约束，isam需要进行更多次的优化
        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
        // 将约束和节点信息清空，他们已经被加入到isam中去了，因此这里清空不会影响整个优化
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        // Step 3 取出优化后的当前帧的位姿，再更新到全局变量中
        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        // 取出优化后的最新关键帧位姿
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");
        // 平移信息取出来保存进cloudKeyPoses3D这个结构中，其中索引作为intensity值
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);
        // 6D姿态同样保留下来
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
        thisPose6D.roll = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // 保存当前位姿的置信度
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

        // save updated transform
        // 将优化后的位姿更新到transformTobeMapped数组中，作为当前最佳估计值
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();


        // save all the received edge and surf points
        // Step 4 存储当前帧的特征点云
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        
        // 当前帧的点云的角点和面点分别拷贝一下
        //! 疑问： 为什么要拷贝呢？反正存储的也是点云的值，直接把这个点云push_back不行吗?
        pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

        // save key frame cloud
        // 关键帧的点云保存下来
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        // Step 5 把优化后的位姿更新到rviz消息队列中，用于rviz显示
        // 根据当前最新位姿更新rviz可视化
        updatePath(thisPose6D);
    }

    /**
     * @brief 如果存在回环或GPS，则关键帧位姿经过优化后会发生变化。这里就把优化后的位姿从因子图中取出来，
     *  更新到全局变量中
     */
    void correctPoses()
    {
        // 没有关键帧，自然也没有什么意义
        if (cloudKeyPoses3D->points.empty())
            return;

        // 只有回环以及gps信息这些会触发全局调整信息才会触发
        if (aLoopIsClosed == true)
        {
            // clear map cache
            // 很多位姿会变化，因此之前的容器内转到世界坐标系下的很多点云就需要调整，因此这里索性清空
            laserCloudMapContainer.clear();
            // clear path
            // 清空path
            globalPath.poses.clear();
            // update key poses
            // 然后更新所有的位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                // 更新所有关键帧的位姿
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();
                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
               
                // 同时更新rviz显示的path
                updatePath(cloudKeyPoses6D->points[i]);
            }
            // 标志位置位
            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose &pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    /**
     * @brief  
     *  1. 发布包含回环优化的全局位姿里程计消息，注意这个是没有累计漂移的
     *  2. 发布上述消息的tf变换，这个rviz会用到它进行位姿、点云等的显示，所以必须发布
     *  3. 发布给前端IMU预积分矫正零偏使用的lidar增量里程计，这个是利用scan-to-map优化的增量，叠加到上一次
     *     的lidar位姿上，所以得到的是纯增量的lidar里程计，不包括回环，是有累积漂移的。但是目的就是要这样的
     *     平滑的里程计位姿，因为一旦包含了回环信息之后，位姿会有突变，会导致前端IMU预积分的因子图优化崩溃
     */
    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        // Step 1 发送当前帧的全局位姿，注意是无全局漂移的
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);

        // Publish TF
        // Step 2 发送lidar在odom坐标系下的tf
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        // Step 3 发送增量里程计给前端IMU预积分矫正零偏使用，这里发布的是增量的平滑里程计，有全局漂移
        // 这里主要用于给imu预积分模块使用，需要里程计是平滑的
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine;         // incremental odometry in affine
        // 该标志位处理一次后始终为true
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            // 记录当前位姿
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        }
        else
        {
            // 上一帧的最佳位姿和当前帧最佳位姿（scan matching之后，而不是根据回环或者gps调整之后的位姿）之间的位姿增量
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            // 位姿增量叠加到上一帧位姿上
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            // 分解成欧拉角+平移向量
            pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);
            // 如果有imu信号，同样对roll和pitch做插值
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            // 协方差这一位作为是否退化的标志位
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    /**
     * @brief 发布各种rviz显示信息
     */
    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        // Step 1 发布关键帧位置
        publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames

        // Step 2 发布用于本次匹配的局部地图点云
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            // 把当前点云转换到世界坐标系下去
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
            // 发送当前点云
            publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }

        // publish registered high-res raw cloud
        // Step 3 发布当前帧的原始点云
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }

        // publish path
        // Step 4 发布历史上所有帧的path信息 
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lio_sam");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

    //; 回环检测线程
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);

    //; rviz可视化相关的线程
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    //! 疑问：看这种写法，把两个线程的加入函数写到ros::spin()后面，那这样还能被调用吗？
    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
