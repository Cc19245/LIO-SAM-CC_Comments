# Cloud Info
Header header 

#; 每个scan上，提取特征点的开始的点的位置 和 结束的点的位置
int32[] startRingIndex
int32[] endRingIndex

# 水平扫描在1800的哪个竖直线上
int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 

#; 是否可以使用IMU的姿态对LiDAR点云去畸变
int64 imuAvailable   

#; 是否可以使用IMU预积分节点发来的里程计位姿，给后端LiDAR优化作为位姿初值
int64 odomAvailable

# Attitude for LOAM initialization
#; 用于整个LIO系统初始化的姿态，从九轴IMU中直接读取得到的
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
#; 用于后端LiDAR的scan-to-map优化的初值，是IMU预积分节点预测的里程计位姿
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature