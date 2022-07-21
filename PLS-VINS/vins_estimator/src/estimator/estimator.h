/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"

#include "../utility/utility.h"
#include "../utility/tic_toc.h"


#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"

#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../factor/projection_factor.h"
#include "../factor/line_parameterization.h"
#include "../factor/line_projection_factor.h"


#include "../featureTracker/feature_tracker.h"
#include "../featureTracker/linefeature_tracker.h"



class Estimator
{
  public:
    Estimator();//构造函数
    ~Estimator();//析构函数
    void setParameter();

    // interface 接口
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame, const map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &line_featureFrame);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &lines, const double header);
    void processMeasurements();
    void changeSensorType(int use_imu, int use_stereo);

    // internal 内部
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    // void optimization();
    void optimizationwithLine();
    void onlyLineOpt();
    void vector2double();
    void double2vector();
    void double2vector2();//线特征添加的
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                              vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);
    // cv::Mat drawPointLine(drawData data);

    // VINS系统的两种状态：
    enum SolverFlag
    {
        INITIAL,// 还未成功初始化
        NON_LINEAR // 已成功初始化，正处于紧耦合优化状态
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,//边缘化老的帧
        MARGIN_SECOND_NEW = 1//边缘化新的帧
    };

    

    double frame_cnt_ = 0;
    double sum_solver_time_ = 0.0;
    double mean_solver_time_ = 0.0;
    double sum_marg_time_ = 0.0;
    double mean_marg_time_=0.0;

    std::mutex mProcess;
    std::mutex mBuf;
    std::mutex mPropagate;
    queue<pair<double, Eigen::Vector3d>> accBuf;//加速度计队列
    queue<pair<double, Eigen::Vector3d>> gyrBuf;//陀螺仪队列
    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 4, 1> > > > > > line_featureBuf;
    double prevTime, curTime;//这两个是按照哪个传感器的时间算的？ IMU时间，因为已经通过dt把相机时间转换到了IMU时间
    bool openExEstimation;

    std::thread trackThread;//没有使用
    std::thread processThread;//处理测量值的线程

    FeatureTracker featureTracker;
    LineFeatureTracker lineFeatureTracker;
    SolverFlag solver_flag;//是否初始化标志位
    MarginalizationFlag  marginalization_flag;//边缘化哪一帧的标志位
    Vector3d g;//重力加速度
    
    Matrix3d ric[2];//相机到imu的旋转矩阵
    Vector3d tic[2];//相机到imu的平移

    Vector3d        Ps[(WINDOW_SIZE + 1)];//划窗内所有的p
    Vector3d        Vs[(WINDOW_SIZE + 1)];//划窗内所有的速度
    Matrix3d        Rs[(WINDOW_SIZE + 1)];//划窗内所有的R
    Vector3d        Bas[(WINDOW_SIZE + 1)];//划窗内所有的bias of a
    Vector3d        Bgs[(WINDOW_SIZE + 1)];//划窗内所有的bias of g
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)]; //窗口内所有帧的时间

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];//里边放的是imu的预积分
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];//加速度的预积分值,放的是划窗内每两帧之间的预积分值
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];//gyro的预积分值

    int frame_count; //窗口内的第几帧,最大值为WINDOW_SIZE + 1
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt; //一共输入了多少图片

    FeatureManager f_manager;//FIXME:定义一个管理特征点的对象
    MotionEstimator m_estimator;//定义一个运动估计的对象
    InitialEXRotation initial_ex_rotation;//定义一个估计外部参数校准的对象

    bool first_imu;//该图像之后的第一个imu
    bool is_valid, is_key;//FIXME: 没有用到
    bool failure_occur;//检测是否发生了错误,在failureDetection中

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;//里边存放关键帧的位姿
    double initial_timestamp;//初始时间戳

    double baseline_;//基线长度
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];//11*7,放了划窗内帧的位姿
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_LineFeature[NUM_OF_F][SIZE_LINE];
    double para_Ex_Pose[2][SIZE_POSE];//相机外部参数的位姿 2*7 两个相机的位姿
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;//上一时刻的先验信息,也就是上一个H矩阵matg掉一部分后剩下的内容
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;//这个是输入到图像中的预积分值

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;// 上一时刻的各个值
    Eigen::Quaterniond latest_Q;

    bool initFirstPoseFlag; //IMU初始位姿的flag
    bool initThreadFlag;//FIXME: esimator构造函数初始化为false
    

    //relocalization variable 线特征添加的
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
