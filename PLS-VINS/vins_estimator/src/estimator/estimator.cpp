/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator() : f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

//所有的状态清空
void Estimator::clearState()
{
    mProcess.lock();
    while (!accBuf.empty())
        accBuf.pop();
    while (!gyrBuf.empty())
        gyrBuf.pop();
    while (!featureBuf.empty())
        featureBuf.pop();
    while (!line_featureBuf.empty())
        line_featureBuf.pop();
    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
    // cout<<"clearstate after"<<endl;
}

// 设置参数，并开启processMeasurements线程
void Estimator::setParameter()
{
    mProcess.lock(); //涉及到多线程
    // 讲相机参数传入
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl
             << ric[i] << endl
             << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric); //将相机参数传入特征点的管理器类中

    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity(); // Matrix2d::Identity() 2*2的单位矩阵
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    baseline_ = BASELINE;
    td = TD; //时间的误差量
    g = G;   //理想的重力加速度
    cout << "set g " << g.transpose() << endl;
    // 将相机参数传入到特征跟踪的类里  set g 0 0 9.81007
    featureTracker.readIntrinsicParameter(CAM_NAMES); //读取相机的内部参数，CAM_NAMES是相机参数的路径

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    // MULTIPLE_THREAD is 1
    if (MULTIPLE_THREAD && !initThreadFlag)
    // 如果是单线程，且线程没有chuli则开启开启了一个Estimator类内的新线程：processMeasurements();
    {
        initThreadFlag = true;
        //申明并定义一个 处理 的线程 自动运行
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if (!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if (USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if (USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }

        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if (restart)
    {
        clearState();
        setParameter();
    }
}

cv::Mat drawPointLine(drawData data)
{
    cv::Mat imTrack;
    int rows = data.img.rows;
    int cols = data.img.cols;

    // ------------将两幅图像进行拼接
    if (!data.img1.empty() && STEREO)
        cv::hconcat(data.img, data.img1, imTrack);
    // 图像凭借hconcat（B,C，A）; // 等同于A=[B  C]
    else
        imTrack = data.img.clone();
    cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);
    //将imTrack转换为彩色

    // -------------在左目图像上标记特征点   cv::Scalar的构造函数是cv::Scalar(v1, v2, v3, v4)，前面的三个参数是依次设置BGR的，和RGB相反,第四个参数设置图片的透明度
    for (size_t j = 0; j < data.cur_pts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * data.track_cnt[j] / 20); // FIXME: 这个是画圈的颜色问题
        // cout<<"track_cnt:"<<track_cnt[j]<<endl;
        cv::circle(imTrack, data.cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 8); //点越红说明被越多帧看到，越蓝说明被很少的帧一起看到
    }

    // -------------在右目图像上标记特征点
    if (!data.img1.empty() && STEREO)
    {
        for (size_t i = 0; i < data.cur_right_pts.size(); i++)
        {
            cv::Point2f rightPt = data.cur_right_pts[i];
            rightPt.x += cols;                                         //计算在凭借的图像上，右目特征点的位置
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 8); //点是绿色的
        }
    }

    //左目画线
    for (int k = 0; k < data.keylsd.size(); ++k)
    {
        unsigned int r = 255; // lowest + int(rand() % range);
        unsigned int g = 255; // lowest + int(rand() % range);
        unsigned int b = 0;   // lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(data.keylsd[k].startPointX), int(data.keylsd[k].startPointY));
        cv::Point endPoint = cv::Point(int(data.keylsd[k].endPointX), int(data.keylsd[k].endPointY));
        cv::line(imTrack, startPoint, endPoint, cv::Scalar(b, g, r), 2, 8); //线是黄色的
    }

    //右目画线
    for (int k = 0; k < data.right_keylsd.size(); ++k)
    {
        unsigned int r = 255; // lowest + int(rand() % range);
        unsigned int g = 255; // lowest + int(rand() % range);
        unsigned int b = 0;   // lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(data.right_keylsd[k].startPointX) + cols, int(data.right_keylsd[k].startPointY));
        cv::Point endPoint = cv::Point(int(data.right_keylsd[k].endPointX) + cols, int(data.right_keylsd[k].endPointY));
        cv::line(imTrack, startPoint, endPoint, cv::Scalar(b, g, r), 2, 8); //线是黄色的
    }

    for (int i = 0; i < data.match_keylsd.size(); i++)
    {
        unsigned int r = 255; // lowest + int(rand() % range);
        unsigned int g = 0;   // lowest + int(rand() % range);
        unsigned int b = 0;   // lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(data.match_keylsd[i].startPointX) + cols, int(data.match_keylsd[i].startPointY));
        cv::Point endPoint = cv::Point(int(data.match_keylsd[i].endPointX) + cols, int(data.match_keylsd[i].endPointY));
        cv::line(imTrack, startPoint, endPoint, cv::Scalar(b, g, r), 2, 8); //线是红色的
    }

    // map<int, cv::Point2f>::iterator mapIt;
    // for (size_t i = 0; i < curLeftIds.size(); i++)
    // {
    //     int id = curLeftIds[i];
    //     mapIt = prevLeftPtsMap.find(id);
    //     if(mapIt != prevLeftPtsMap.end())
    //     {
    //         cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    //         // 在imTrack上，从curLeftPts到mapIt->second画箭头
    //     }
    //}
    return imTrack;
}

// 给Estimator输入图像
// 其实是给featureTracker.trackImage输入图像，之后返回图像特征featureFrame。填充featureBuf
// 之后执行processMeasurements
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    //直方圖均衡化
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8)); // createCLAHE 直方图均衡
    clahe->apply(_img, _img);
    if (!_img1.empty())
        clahe->apply(_img1, _img1);

    inputImageCnt++; //计算输入的图片数量
    /**************线特征有关***************/

    // queue<sensor_msgs::ImageConstPtr> img_buf;

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    // 数据格式为feature_id camera_id（0或1） xyz_uv_velocity（空间坐标，像素坐标和像素速度）
    map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> lineFeatureFrame;
    //数据格式为feature_id camera_id（0或1） start_end_xy（起点终点的归一化平面坐标）
    drawData data; //存储点线信息用来画图

    if (_img1.empty())
    {
        TicToc featureTrackerTime; // 點特征追踪所用的时间 这个很好用啊！！
        // thread pointTrack(&FeatureTracker::trackImage, &featureTracker, &featureFrame, &data, t, _img, cv::Mat());

        featureTracker.trackImage(&featureFrame, &data, t, _img); // 追踪单目
        printf("point_FeatureTracker time: %f\n", featureTrackerTime.toc());
        if (NUM_OF_LINE == 1)
        {
            TicToc lineFeatureTrackerTime; //线特征追踪所用的时间
            // thread lineTrack(&LineFeatureTracker::readImage, &lineFeatureTracker, &lineFeatureFrame, &data, featureTracker, t, _img,cv::Mat());
            // lineTrack.join();

            lineFeatureTracker.readImage(&lineFeatureFrame, &data, featureTracker, t, _img); //追踪单目线特征
            // printf("zj line_FeatureTracker time: %f\n", lineFeatureTrackerTime.toc());
        }
        // pointTrack.join();
    }
    else
    {

        // thread pointTrack(&FeatureTracker::trackImage, &featureTracker, &featureFrame, &data, t, _img, _img1);
        // pointTrack.join();
        TicToc featureTrackerTime;                                       // 点特征追踪所用的时间 这个很好用啊！！
        featureTracker.trackImage(&featureFrame, &data, t, _img, _img1); // 追踪双目
        // printf("point_FeatureTracker time: %f\n", featureTrackerTime.toc());
        if (NUM_OF_LINE == 1)
        {
            // thread lineTrack(&LineFeatureTracker::readImage, &lineFeatureTracker, &lineFeatureFrame, &data, featureTracker, t, _img,cv::Mat());
            // lineTrack.join();
            TicToc lineFeatureTrackerTime;                                                   //线特征追踪所用的时间
            lineFeatureTracker.readImage(&lineFeatureFrame, &data, featureTracker, t, _img); //追踪单目线特征
            printf("line_FeatureTracker time: %f\n", lineFeatureTrackerTime.toc());
        }
        if (NUM_OF_LINE == 2)
        {
            // thread lineTrack(&LineFeatureTracker::readImage, &lineFeatureTracker, &lineFeatureFrame, &data, featureTracker, t, _img, _img1);
            // lineTrack.join();
            TicToc lineFeatureTrackerTime;                                                          //线特征追踪所用的时间
            lineFeatureTracker.readImage(&lineFeatureFrame, &data, featureTracker, t, _img, _img1); //追踪双目线特征
            // printf("line_FeatureTracker time: %f\n", lineFeatureTrackerTime.toc());
        }
    }

    if (SHOW_TRACK) //这个应该是展示轨迹
    {
        cv::Mat imgTrack = drawPointLine(data);
        // cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }

    if (MULTIPLE_THREAD) //多线程的话输入两张图片处理一次
    {
        if (inputImageCnt % 1 == 0)
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            if (NUM_OF_LINE >= 1)
                line_featureBuf.push(make_pair(t, lineFeatureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        if (NUM_OF_LINE >= 1)
            line_featureBuf.push(make_pair(t, lineFeatureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements(); //这里才执行了processMeasurements这个线程
        printf("process time: %f\n", processTime.toc());
    }
}

// 输入一个imu的量测
// 填充了accBuf和gyrBuf
void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    // printf("input imu with time %f \n", t);
    mBuf.unlock();

    // 如果已经初始化了
    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

// 在估计器中输入点云数据，并填充featureBuf,line_featureBuf
void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame, const map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &line_featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    // line_featureBuf.push(make_pair(t,line_featureFrame));
    mBuf.unlock();

    if (!MULTIPLE_THREAD)
        processMeasurements();
}

// 对imu的时间进行判断，讲队列里的imu数据放入到accVector和gyrVector中，完成之后返回true
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                               vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if (accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    // printf("get imu from %f %f\n", t0, t1);
    // printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);

    // 当现在时间小于队列里最后一个时间时
    if (t1 <= accBuf.back().first)
    {
        //如果队列里第一个数据的时间小于起始时间，则删除第一个元素
        while (accBuf.front().first <= t0)
        {
            accBuf.pop(); //.pop删除栈顶元素
            gyrBuf.pop();
        }
        // 讲队列里所有的acc和gyr输入到accvector个gyrvector中
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

// 判断输入的时间t时候的imu是否可用
bool Estimator::IMUAvailable(double t)
{
    if (!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

//处理量测的线程
void Estimator::processMeasurements()
{
    while (1)
    {
        // printf("process measurments\n");
        // featurede 结构:
        //     时间       特征id           相机编号(0/1)         观测值
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature;
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>>> line_feature;
        // 时间 特征点ID 图像id xyz_uv_vel

        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        //处理特征点的buf
        if (!featureBuf.empty() || !line_featureBuf.empty())
        {
            feature = featureBuf.front(); //.front()返回当前vector容器中起始元素的引用。
            curTime = feature.first + td; // td的使用是在图像的时间上加上这个值
            if (!line_featureBuf.empty())
                line_feature = line_featureBuf.front();

            while (1)
            {
                if ((!USE_IMU || IMUAvailable(feature.first + td))) //如果不用imu或者imu不可用
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (!MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5); //定义5ms的延迟
                    std::this_thread::sleep_for(dura); //这个线程延迟5ms
                }
            }
            mBuf.lock();
            if (USE_IMU)
                // 对imu的时间进行判断，讲队列里的imu数据放入到accVector和gyrVector中，完成之后返回true
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            featureBuf.pop(); //每次运行完之后都删除featureBuf中的元素，直到为空，已经把要删除的这个值给了feature
            if (!line_featureBuf.empty())
            {
                line_featureBuf.pop();
            }
            mBuf.unlock();

            // 处理imu数据，运行processIMU
            if (USE_IMU)
            {
                if (!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                for (size_t i = 0; i < accVector.size(); i++)
                {
                    double dt; //计算每次imu量测之间的dt
                    if (i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;

                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }
            mProcess.lock();
            processImage(feature.second, line_feature.second, feature.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }
        //     else if(NUM_OF_LINE==0 && !featureBuf.empty() )
        //     {
        //         feature = featureBuf.front();//.front()返回当前vector容器中起始元素的引用。
        //         curTime = feature.first + td;//td的使用是在图像的时间上加上这个值
        //         // if(line_featureBuf.empty())
        //         // {
        //         //     cout<<"no line"<<endl;
        //         //     getchar();
        //         // }
        //         line_feature=line_featureBuf.front();

        //         while(1)
        //         {
        //             if ((!USE_IMU  || IMUAvailable(feature.first + td)))//如果不用imu或者imu不可用
        //                 break;
        //             else
        //             {
        //                 printf("wait for imu ... \n");
        //                 if (! MULTIPLE_THREAD)
        //                     return;
        //                 std::chrono::milliseconds dura(5);//定义5ms的延迟
        //                 std::this_thread::sleep_for(dura);//这个线程延迟5ms
        //             }
        //         }
        //         mBuf.lock();
        //         if(USE_IMU)
        //             // 对imu的时间进行判断，讲队列里的imu数据放入到accVector和gyrVector中，完成之后返回true
        //             getIMUInterval(prevTime, curTime, accVector, gyrVector);

        //         featureBuf.pop();//每次运行完之后都删除featureBuf中的元素，直到为空，已经把要删除的这个值给了feature
        //         if(!line_featureBuf.empty())
        //         {
        //             line_featureBuf.pop();
        //             cout<<"line pop()"<<endl;
        //         }
        //         mBuf.unlock();

        //         // 处理imu数据，运行processIMU
        //         if(USE_IMU)
        //         {
        //             if(!initFirstPoseFlag)
        //                 initFirstIMUPose(accVector);
        //             for(size_t i = 0; i < accVector.size(); i++)
        //             {
        //                 double dt;//计算每次imu量测之间的dt
        //                 if(i == 0)
        //                     dt = accVector[i].first - prevTime;
        //                 else if (i == accVector.size() - 1)
        //                     dt = curTime - accVector[i - 1].first;
        //                 else
        //                     dt = accVector[i].first - accVector[i - 1].first;

        //                 processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
        //             }
        //         }
        //         mProcess.lock();
        //         processImage(feature.second, line_feature.second, feature.first);
        //         prevTime = curTime;

        //         printStatistics(*this, 0);

        //         std_msgs::Header header;
        //         header.frame_id = "world";
        //         header.stamp = ros::Time(feature.first);

        //         pubOdometry(*this, header);
        //         pubKeyPoses(*this, header);
        //         pubCameraPose(*this, header);
        //         pubPointCloud(*this, header);
        //         pubKeyframe(*this);
        //         pubTF(*this, header);
        //         mProcess.unlock();
        //     }

        if (!MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

//初始第一个imu位姿
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    // return;
    //计算加速度的均值
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for (size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());

    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    // cout<<"second yaw:"<<yaw<<endl;
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; //旋转两次 另初始的航向为0
    Rs[0] = R0;
    cout << "init R0 " << endl
         << Rs[0] << endl;
    // Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

/* 对imu计算预积分
传进来的是一个imu数据 得到预积分值pre_integrations 还有一个tmp_pre_integration */
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // 第一个imu处理
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 如果是新的一帧,则新建一个预积分项目
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    // frame_count是窗内图像帧的计数
    //  一个窗内有十个相机帧，每个相机帧之间又有多个IMU数据
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        // push_back进行了重载，的时候就已经进行了预积分

        // if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity); //这个是输入到图像中的预积分值

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // 对位移速度等进行累加
        // Rs Ps Vs是frame_count这一个图像帧开始的预积分值,是在绝对坐标系下的.
        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;            //移除了偏执的加速度
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j]; //移除了偏执的gyro
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    // 让此时刻的值等于上一时刻的值，为下一次计算做准备
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// 其中包含了检测关键帧,估计外部参数,初始化,状态估计,划窗等等
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &lines, const double header)
// image 里边放的就是该图像的特征点 header 时间
{
    ROS_DEBUG("new image coming ------------------------------------------"); //输入进来的其实只有特征点
    ROS_DEBUG("Adding feature points %lu", image.size());

    // 检测关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, lines, td)) // 当视差较大时，marg 最老的关键帧
    {
        marginalization_flag = MARGIN_OLD; //新一帧将被作为关键帧!
        // printf("keyframe\n");
    }
    else // 当视差较小时，比如静止，marg 倒数第二的图像帧
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        // printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of point feature: %d", f_manager.getFeatureCount());
    ROS_DEBUG("number of line feature: %d", f_manager.getLineFeatureCount());

    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // 估计一个外部参,并把ESTIMATE_EXTRINSIC置1,输出ric和RIC
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            // 这个里边放的是新图像和上一帧
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                               << calib_ric);
                //有几个相机，就有几个ric，目前单目情况下，ric内只有一个值
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // 这里进行初始化
    if (solver_flag == INITIAL)
    {

        // 单目初始化 monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;

                //有外参且当前帧时间戳大于初始化时间戳0.1秒，就进行初始化操作
                if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1) // initial_timestamp设为了0
                {
                    result = initialStructure(); //视觉惯性联合初始化
                    initial_timestamp = header;  //更新初始化时间戳
                }
                if (result) //如果初始化成功
                {
                    // f_manager.debugShow();
                    //先进行一次滑动窗口非线性优化，得到当前帧与第一帧的位姿
                    // optimization();
                    onlyLineOpt(); // 三角化以后，优化一把
                    optimizationwithLine();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow(); //滑动窗口
                    f_manager.removeFailures();
                    ROS_INFO("Initialization finish!");
                }
                else //滑掉一帧
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        if (STEREO && USE_IMU)
        {
            TicToc init_time;
            // 双目pnp求解出滑窗内所有相机姿态，三角化特征点空间位置。
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric); //有了深度就可以进行求解了
            TicToc t_tri;
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulateLine(frame_count, Ps, Rs, tic, ric);
            // cout<<"triangulation costs :"<<t_tri.toc()<<endl;
            ROS_DEBUG("come here %d", frame_count);
            // 将结果放入到队列当中
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }

                solveGyroscopeBias(all_image_frame, Bgs);
                // 对之前预积分得到的结果进行更新。
                // 预积分的好处查看就在于你得到新的Bgs，不需要又重新再积分一遍，可以通过Bgs对位姿，速度的一阶导数，进行线性近似，得到新的Bgs求解出MU的最终结果。
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                onlyLineOpt(); // 三角化以后，优化一把
                optimizationwithLine();
                // optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
                cout << "init time:" << init_time.toc() << endl;
                cout << "双目初始化完成" << endl;
            }
        }

        // stereo only initilization
        if (STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            onlyLineOpt(); // 三角化以后，优化一把
            optimizationwithLine();

            if (frame_count == WINDOW_SIZE)
            {
                onlyLineOpt(); // 三角化以后，优化一把
                optimizationwithLine();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // 如果划窗内的没有算法,进行状态更新
        if (frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }
    }

    // 如果已经进行了初始化
    else
    {
        TicToc t_solve; //优化所用的时间
        if (!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric); //直接对下一帧求解位姿
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulateLine(frame_count, Ps, Rs, tic, ric);
        onlyLineOpt(); // 三角化以后，优化一把
        optimizationwithLine();
        // f_manager.debugShow();
        // optimization();
        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);

        // FIXME: 如果不是多线程,就进行预测??
        if (!MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }

        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!"); //系统重启
            return;
        }

        slideWindow();
        f_manager.removeFailures();

        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        // 对划窗的状态进行更新,记录上一次划窗内的初始和最后的位姿
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }
}

/* 视觉的结构初始化
首先得到纯视觉的,所有图像在IMU坐标系下的,一个初始化结果,也就是RT
然后进行视觉imu对其,陀螺仪偏执估计等等
其中包含 */
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // check imu observibility
    //  通过计算线加速度的标准差，是否小于0.25 判断IMU是否有充分运动激励，以进行初始化
    //  注意这里并没有算上all_image_frame的第一帧，所以求均值和标准差的时候要减一
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        // ROS_WARN("IMU variation %f!", var);
        if (var < 0.25) //对标准差进行判断
        {
            ROS_INFO("IMU excitation not enougth!");
            // return false;
        }
    }
    // global sfm

    // 将f_manager中的所有feature保存到vector<SFMFeature> sfm_f中
    // 这里解释一下SFMFeature，其存放的是特征点的信息
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;

    //     struct SFMFeature 其存放的是特征点的信息
    // {
    //     bool state;//状态（是否被三角化）
    //     int id;
    //     vector<pair<int,Vector2d>> observation;//所有观测到该特征点的图像帧ID和图像坐标
    //     double position[3];//3d坐标
    //     double depth;//深度
    // };
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;

    // 保证具有足够的视差,由E矩阵恢复R、t
    // 这里的第L帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用，得到的Rt为当前帧到第l帧的坐标系变换Rt
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    // 对窗口中每个图像帧求解sfm问题，得到所有图像帧相对于参考帧的旋转四元数Q、平移向量T和特征点坐标sfm_tracked_points。
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // 对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化
    // solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        //注意这里的 Q和 T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵，两者具有对称关系
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        //罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        //获取 pnp需要用到的存储每个特征点三维点和图像坐标的 vector
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        //保证特征点数大于 5
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        /**
         *bool cv::solvePnP(    求解 pnp问题
         *   InputArray  objectPoints,   特征点的3D坐标数组
         *   InputArray  imagePoints,    特征点对应的图像坐标
         *   InputArray  cameraMatrix,   相机内参矩阵
         *   InputArray  distCoeffs,     失真系数的输入向量
         *   OutputArray     rvec,       旋转向量
         *   OutputArray     tvec,       平移向量
         *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
         *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
         *)
         */
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        //罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        //这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    /* visualInitialAlign
    很具VIO课程第七讲:一共分为5步:
    1估计旋转外参. 2估计陀螺仪bias 3估计重力方向,速度.尺度初始值 4对重力加速度进一步优化 5将轨迹对其到世界坐标系 */
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

// 视觉和惯性的对其,对应https://mp.weixin.qq.com/s/9twYJMOE8oydAzqND0UmFw中的visualInitialAlign
/* visualInitialAlign
很具VIO课程第七讲:一共分为5步:
1估计旋转外参. 2估计陀螺仪bias 3估计重力方向,速度.尺度初始值 4对重力加速度进一步优化 5将轨迹对其到世界坐标系 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    // solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // 初始化完成之后,对所有状态进行更改change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    // //以下为线特征修改的
    // VectorXd dep = f_manager.getDepthVector();
    // for (int i = 0; i < dep.size(); i++)
    //     dep[i] = -1;
    // f_manager.clearDepth(dep);

    // //triangulat on cam pose , no tic
    // Vector3d TIC_TMP[NUM_OF_CAM];
    // for(int i = 0; i < NUM_OF_CAM; i++)
    //     TIC_TMP[i].setZero();
    // ric[0] = RIC[0];
    // f_manager.setRic(ric);
    // f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));
    // //以上为线特征修改的

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    //所有变量从c0系旋转到w系
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
    f_manager.triangulateLine(frame_count, Ps, Rs, tic, ric);

    // cout<<"triangulate after"<<endl;
    return true;
}

// 该函数判断每帧到窗口最后一帧对应特征点的平均视差大于30，且内点数目大于12则可进行初始化，同时返回当前帧到第l帧的坐标系变换R和T
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        //寻找第i帧到窗口最后一帧的对应特征点
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);

        //计算平均视差
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;

            //第j个对应点在第i帧和最后一帧的(x,y)
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());

            //判断是否满足初始化条件：视差>30和内点数满足要求(大于12)
            // solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
            //同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的relative_R，relative_T
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

// vector转换成double数组，因为ceres使用数值数组
/*可以看出来，这里面生成的优化变量由：
para_Pose（7维，相机位姿）、
para_SpeedBias（9维，相机速度、加速度偏置、角速度偏置）、
para_Ex_Pose（7维、相机IMU外参）、
para_Feature（1维，特征点深度）、
para_Td（1维，标定同步时间）
五部分组成，在后面进行边缘化操作时这些优化变量都是当做整体看待。*/
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if (USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;

#ifdef LINEINCAM
    MatrixXd lineorth = f_manager.getLineOrthVectorInCamera();
#else
    MatrixXd lineorth = f_manager.getLineOrthVector(Ps, Rs, tic, ric);
#endif

    for (int i = 0; i < f_manager.getLineFeatureCount(); ++i)
    {
        para_LineFeature[i][0] = lineorth.row(i)[0];
        para_LineFeature[i][1] = lineorth.row(i)[1];
        para_LineFeature[i][2] = lineorth.row(i)[2];
        para_LineFeature[i][3] = lineorth.row(i)[3];
        if (i > NUM_OF_F)
            std::cerr << " 1000  1000 1000 1000 1000 \n\n";
    }
}

// 数据转换，vector2double的相反过程
void Estimator::double2vector()
{
    // 相机姿态需要变化考虑优化以后，把yaw量旋转回去
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]); //优化之前的0th的姿态
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if (USE_IMU)
    {
        // 优化以后的0th的姿态
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                         para_Pose[0][3],
                                                         para_Pose[0][4],
                                                         para_Pose[0][5])
                                                 .toRotationMatrix());
        // 优化前后，yaw的变化
        double y_diff = origin_R0.x() - origin_R00.x();
        // TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5])
                                   .toRotationMatrix()
                                   .transpose();
        }

        // 由于VI系统的（绝对位置x,y,z,以及yaw）是不可观的。而优化过程中没有固定yaw角，因此yaw会朝着使得误差函数最小的方向优化，但这不一定是正确的。
        // 所以这里把 yaw角的变化量给旋转回去。
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            // Position 也转移到yaw角优化前的 0th坐在的世界坐标下
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                        para_Pose[i][1] - para_Pose[0][1],
                                        para_Pose[i][2] - para_Pose[0][2]) +
                    origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                              para_SpeedBias[i][4],
                              para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                              para_SpeedBias[i][7],
                              para_SpeedBias[i][8]);
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if (USE_IMU)
    {
        // 跟yaw没关系，所以不用管优化前后yaw的变化
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5])
                         .toRotationMatrix();
        }
    }

    // std::cout <<"----------\n"<< Rwow1 <<"\n"<<twow1<<std::endl;
    MatrixXd lineorth_vec(f_manager.getLineFeatureCount(), 4);
    for (int i = 0; i < f_manager.getLineFeatureCount(); ++i)
    {
        Vector4d orth(para_LineFeature[i][0],
                      para_LineFeature[i][1],
                      para_LineFeature[i][2],
                      para_LineFeature[i][3]);
        lineorth_vec.row(i) = orth;
    }
#ifdef LINEINCAM
    f_manager.setLineOrthInCamera(lineorth_vec);
#else
    f_manager.setLineOrth(lineorth_vec, Ps, Rs, tic, ric);
#endif

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if (USE_IMU)
        td = para_Td[0][0];
}

void Estimator::double2vector2()
{
    // 六自由度优化的时候，整个窗口会在空间中任意优化，这时候我们需要把第一帧在yaw,position上的增量给去掉，因为vins在这几个方向上不可观，他们优化的增量也不可信。
    // 所以这里的操作过程就相当于是 fix 第一帧的 yaw 和 postion, 使得整个轨迹不会在空间中任意飘。
    // 相机姿态需要变化考虑优化以后，把yaw量旋转回去
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]); //优化之前的0th的姿态
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Matrix3d rot_diff_bf;
    if (USE_IMU)
    {
        // 优化以后的0th的姿态
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                         para_Pose[0][3],
                                                         para_Pose[0][4],
                                                         para_Pose[0][5])
                                                 .toRotationMatrix());

        // 优化前后，yaw的变化
        double y_diff = origin_R0.x() - origin_R00.x();
        // TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        rot_diff_bf = rot_diff;
        // 由于VI系统的（绝对位置x,y,z,以及yaw）是不可观的。而优化过程中没有固定yaw角，因此yaw会朝着使得误差函数最小的方向优化，但这不一定是正确的。
        // 所以这里把 yaw角的变化量给旋转回去。
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            // Position 也转移到yaw角优化前的 0th坐在的世界坐标下
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                        para_Pose[i][1] - para_Pose[0][1],
                                        para_Pose[i][2] - para_Pose[0][2]) +
                    origin_P0;
            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                              para_SpeedBias[i][4],
                              para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                              para_SpeedBias[i][7],
                              para_SpeedBias[i][8]);
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if (USE_IMU)
    {
        // 跟yaw没关系，所以不用管优化前后yaw的变化
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5])
                         .toRotationMatrix();
        }
    }

    // 先把line旋转到相机坐标系下
    Matrix3d Rwow1 = rot_diff_bf;
    Vector3d tw1b(para_Pose[0][0], para_Pose[0][1], para_Pose[0][2]);
    Vector3d twow1 = -Rwow1 * tw1b + origin_P0;

    // std::cout <<"----------\n"<< Rwow1 <<"\n"<<twow1<<std::endl;
    MatrixXd lineorth_vec(f_manager.getLineFeatureCount(), 4);
    ;
    for (int i = 0; i < f_manager.getLineFeatureCount(); ++i)
    {
        Vector4d orth(para_LineFeature[i][0],
                      para_LineFeature[i][1],
                      para_LineFeature[i][2],
                      para_LineFeature[i][3]);

        // 将line_w优化以后的角度变化yaw的变化旋转回去
        Vector6d line_w1 = orth_to_plk(orth);
        Vector6d line_wo = plk_to_pose(line_w1, Rwow1, twow1);
        orth = plk_to_orth(line_wo);

        lineorth_vec.row(i) = orth;
    }
    f_manager.setLineOrth(lineorth_vec, Ps, Rs, tic, ric);

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if (USE_IMU)
        td = para_Td[0][0];
}

//检测是否发生错误
bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 1)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 1.5) //加速度的偏执大于2了
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0) //陀螺仪的偏执>1
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        return true;
    }
    return false;
}

void Estimator::onlyLineOpt()
{
    //固定pose， 只优化line的参数，用来调试line的一些参数，看ba优化出来的最好line地图是啥样
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++) // 将窗口内的 p,q 加入优化变量
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization); // p,q
        // 固定 pose
        problem.SetParameterBlockConstant(para_Pose[i]);
    }
    for (int i = 0; i < NUM_OF_CAM; i++) // 外参数
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);

        // 固定 外参数
        problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    }
    vector2double(); // 将那些保存在 vector向量里的参数 移到 double指针数组里去

    // 所有特征
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.linefeature)
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();                                                        // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation)) // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
            continue;

        ++feature_index; // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式
        /*
        std::cout << para_LineFeature[feature_index][0] <<" "
                << para_LineFeature[feature_index][1] <<" "
                << para_LineFeature[feature_index][2] <<" "
                << para_LineFeature[feature_index][3] <<"\n";
        */
        ceres::LocalParameterization *local_parameterization_line = new LineOrthParameterization();
        problem.AddParameterBlock(para_LineFeature[feature_index], SIZE_LINE, local_parameterization_line); // p,q

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        for (auto &it_per_frame : it_per_id.linefeature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                // continue;
            }
            Vector4d obs = it_per_frame.lineobs;                     // 在第j帧图像上的观测
            lineProjectionFactor *f = new lineProjectionFactor(obs); // 特征重投影误差
            problem.AddResidualBlock(f, loss_function,
                                     para_Pose[imu_j],
                                     para_Ex_Pose[0],
                                     para_LineFeature[feature_index]);
            f_m_cnt++;
        }
    }

    if (feature_index < 3)
    {
        return;
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // std::cout <<"!!!!!!!!!!!!!onlyLineOpt!!!!!!!!!!!!!\n";
    double2vector();
    // std::cout << summary.FullReport()<<std::endl;

    f_manager.removeLineOutlier(Ps, Rs, tic, ric);
}

//#define DebugFactor
void Estimator::optimizationwithLine()
{
    frame_cnt_++;

    //------------------ 定义问题 定义本地参数化,并添加优化参数-------------------------------------------------
    ceres::Problem problem;                    // 定义ceres的优化问题
    ceres::LossFunction *loss_function;        //核函数
    loss_function = new ceres::HuberLoss(1.0); // HuberLoss当预测偏差小于 δ 时，它采用平方误差,当预测偏差大于 δ 时，采用的线性误差。
    // loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++) // 将窗口内的 p,q,v,ba,bg加入优化变量
    {
        // 对于四元数或者旋转矩阵这种使用过参数化表示旋转的方式，它们是不支持广义的加法
        // 所以我们在使用ceres对其进行迭代更新的时候就需要自定义其更新方式了，具体的做法是实现一个LocalParameterization
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        // AddParameterBlock   向该问题添加具有适当大小和参数化的参数块。
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization); // p,q 因为有四元数,所以使用了 local_parameterization
        if (USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS); // v,ba,bg  //使用默认的加法
    }

    // 没使用imu时,将窗口内第一帧的位姿固定
    if (!USE_IMU)
        // SetParameterBlockConstant 在优化过程中，使指示的参数块保持恒定。设置任何参数块变成一个常量
        // 固定第一帧的位姿不变!  这里涉及到论文2中的
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++) // 外参数
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization); //如果是双目,估计两个相机的位姿
        // if (!ESTIMATE_EXTRINSIC)
        // {
        //     ROS_DEBUG("fix extinsic param");
        //     problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        // }
        // else
        //     ROS_DEBUG("estimate extinsic param");
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        // Vs[0].norm() > 0.2窗口内第一个速度>2?
        {
            // ROS_INFO("estimate extinsic param");
            openExEstimation = 1; //打开外部估计
        }
        else //如果不需要估计,则把估计器中的外部参数设为定值
        {
            // ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }

    problem.AddParameterBlock(para_Td[0], 1); //把时间也作为待优化变量
    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)   //如果不估计时间就固定
    {
        problem.SetParameterBlockConstant(para_Td[0]);
        // ROS_INFO("fix td");
    }

    TicToc t_whole, t_prepare; // 统计程序运行时间

    vector2double(); // 将那些保存在 vector向量里的参数 移到 double指针数组里去

    // ------------------------在问题中添加约束,也就是构造残差函数----------------------------------
    // 在问题中添加先验信息作为约束
    // 滑动窗口marg以后，上一次的prior factor  添加先验残差，通过Marg的舒尔补操作，将被Marg部分的信息叠加到了保留变量的信息上
    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // 构造新的marginisation_factor construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        /* 通过提供参数块的向量来添加残差块。
        ResidualBlockId AddResidualBlock(
            CostFunction* cost_function,//损失函数
            LossFunction* loss_function,//核函数
            const std::vector<double*>& parameter_blocks); */

        ceres::ResidualBlockId block_id = problem.AddResidualBlock(marginalization_factor, NULL,
                                                                   last_marginalization_parameter_blocks);
        ROS_DEBUG("add marginalization residual");
    }

    // 在问题中添加IMU约束
    if (USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0) // 由于有时候会有静止的情况出现，这时候视差一直不够，关键帧一直没有选，预积分量一直累计，可能出现时间超过10s的情况？
                continue;
            // 前后帧之间建立IMU残差
            IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]); // 预积分误差项: 误差，雅克比的计算
            // 后面四个参数为变量初始值，优化过程中会更新
            ceres::ResidualBlockId block_id = problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
            //这里添加的参数包括状态i和状态j
        }
        ROS_DEBUG("add imu residual");
    }
    // (3) 添加视觉重投影残差
    // 所有特征
    int f_m_cnt = 0; //每个特征点,观测到它的相机帧数
    int feature_index = -1;
    // cout<<"点特征的个数:"<<f_manager.feature.size()<<endl;
    // getchar();
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
        // cout<<"该点特征出现的次数:"<<it_per_id.used_num<<endl;
        if (it_per_id.used_num < 4)
            continue;
        // if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
        //     continue;

        ++feature_index; // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point; // 图像上第一次观测到这个特征的坐标

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j) //既,本次不是第一次观测到 // 非首帧观测帧
            {
                // 当前观测帧归一化相机平面点
                Vector3d pts_j = it_per_frame.point;
                // 首帧与当前观测帧建立重投影误差
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                /* 相关介绍:
                1 只在视觉量测中用了核函数loss_function 用的是huber
                2 参数包含了para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]
                3 ProjectionTwoFrameOneCamFactor这个重投影并不是很懂 */
            }

            // 如果是双目的
            if (STEREO && it_per_frame.is_stereo)
            {
                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j) //既,本次不是第一次观测到 // 首帧与当前观测帧右目建立重投影误差
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else //既,本次是第一次观测到 // 首帧左右目建立重投影误差
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
            }
            f_m_cnt++;
        }
    }
    ROS_DEBUG("add point residual");
    /////////////////////////////////////
    if (NUM_OF_LINE > 0)
    {
        // Line feature
        int line_m_cnt = 0; //每个线特征,观测到它的帧数
        int linefeature_index = -1;
        // cout<<"线特征的个数:"<<f_manager.linefeature.size()<<endl;
        // getchar();
        ROS_DEBUG("zj,number of lines:%d", f_manager.getLineFeatureCount());
        for (auto &it_per_id : f_manager.linefeature)
        {

            it_per_id.used_num = it_per_id.linefeature_per_frame.size();                                                        // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
            if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation)) // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
                continue;

            ++linefeature_index; // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

            // cout<<"add parameter before"<<endl;
            //使用ceres对线特征进行迭代更新的时候就需要自定义其更新方式了，具体的做法是实现一个LocalParameterization
            ceres::LocalParameterization *local_parameterization_line = new LineOrthParameterization();
            // AddParameterBlock   向该问题添加具有适当大小和参数化的参数块。
            problem.AddParameterBlock(para_LineFeature[linefeature_index], SIZE_LINE, local_parameterization_line); // p,q

            // cout<<"add parameter after"<<endl;
            // imu_i该线特征第一次被观测到的帧 ,imu_j = imu_i - 1
            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

            // Vector4d pts_i = it_per_id.linefeature_per_frame[0].lineobs;
            // cout<<"size it_per_id.linefeature_per_frame:"<<it_per_id.linefeature_per_frame.size()<<endl;
            // cout<<"add residual before"<<endl;
            // cout<<it_per_id.linefeature_per_frame.size()<<endl;
            for (auto &it_per_frame : it_per_id.linefeature_per_frame)
            {
                imu_j++;
                // if (imu_i == imu_j)
                // {
                //     //continue;
                // }
                // Vector4d obs = it_per_frame.lineobs;                          // 在第j帧图像上的观测

                // lineProjectionFactor *f = new lineProjectionFactor(obs);     // 特征重投影误差
                // problem.AddResidualBlock(f, loss_function,
                //                          para_Pose[imu_j],
                //                          para_Ex_Pose[0],
                //                          para_LineFeature[linefeature_index]);
                if (imu_i != imu_j)//不是第一次观测
                {
                    Vector4d pts_j = it_per_frame.lineobs;                     // 在第j帧图像上的观测
                    lineProjectionFactor *f = new lineProjectionFactor(pts_j); // 特征重投影误差
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_j], para_Ex_Pose[0], para_LineFeature[linefeature_index]);
                }

                //如果是双目的,并且是双目观测到的
                if (STEREO && it_per_frame.is_stereo && NUM_OF_LINE == 2)
                {
                    // cout<<"stereo in"<<endl;
                    Vector4d pts_j_right = it_per_frame.lineobs_R;
                    if (imu_i != imu_j) //既,本次不是第一次观测到
                    {
                        lineProjectionFactor *f = new lineProjectionFactor(pts_j_right);
                        problem.AddResidualBlock(f, loss_function, para_Pose[imu_j], para_Ex_Pose[1], para_LineFeature[linefeature_index]);
                    }
                    // else//既,本次是第一次观测到
                    // {
                    //     lineProjectionFactor *f = new lineProjectionFactor(pts_j_right);
                    //     problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                    // }
                }
                line_m_cnt++;
            }
            // cout<<"add residual after"<<endl;
        }
        ROS_DEBUG("lineFactor: %d, pointFactor:%d", line_m_cnt, f_m_cnt);
        // ofstream foutC("/home/zj/output/myline.csv", ios::app);
        // foutC<<line_m_cnt<<endl;
    }
    ROS_DEBUG("add line residual");

    ////////////////////////////////////////

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    /***
     * 这一块代码先注释掉，应该和重定位有关
     ***/
    // if(relocalization_info)
    // {
    //     //printf("set relocalization factor! \n");
    //     ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    //     problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
    //     int retrive_feature_index = 0;
    //     int feature_index = -1;
    //     for (auto &it_per_id : f_manager.feature)
    //     {
    //         it_per_id.used_num = it_per_id.feature_per_frame.size();
    //         if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
    //             continue;
    //         ++feature_index;
    //         int start = it_per_id.start_frame;
    //         if(start <= relo_frame_local_index)
    //         {
    //             while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
    //             {
    //                 retrive_feature_index++;
    //             }
    //             if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
    //             {
    //                 Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
    //                 Vector3d pts_i = it_per_id.feature_per_frame[0].point;

    //                 ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
    //                 problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
    //                 retrive_feature_index++;
    //             }
    //         }
    //     }

    // }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = true;

    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;

    TicToc t_solver;
    ceres::Solver::Summary summary; //优化信息
    ceres::Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    ROS_DEBUG("Points Lines Iterations : %d", static_cast<int>(summary.iterations.size()));
    sum_solver_time_ += t_solver.toc();
    mean_solver_time_ = sum_solver_time_ / frame_cnt_;
    ROS_DEBUG("Points Lines solver costs: %f", mean_solver_time_);
    // printf("solver costs: %f \n", t_solver.toc());

    // double2vector();

    double2vector2(); // Line pose change
    TicToc t_culling;
    f_manager.removeLineOutlier(Ps, Rs, tic, ric); // remove Line outlier
    ROS_DEBUG("culling line feautre: %f ms", t_culling.toc());

    if (frame_count < WINDOW_SIZE)
        return;

    // 以下是边缘化操作
    // -----------------------------marginalization ------------------------------------

    TicToc t_whole_marginalization;

    //如果需要marg掉最老的一帧
    if (marginalization_flag == MARGIN_OLD)
    {
        // cout<<"边缘化掉最老的一帧"<<endl;
        // 构建一个新的 prior info
        MarginalizationInfo *marginalization_info = new MarginalizationInfo(); // 将优化以后要marg掉的部分转为prior factor
        vector2double();

        /*
           将最老帧上约束转变为 prior, 那有哪些约束是跟这个最老的帧相关的呢？
           1. 上一次优化以后留下的 prior 里可能存在
           2. 跟最老帧 存在 预积分imu 约束
           3. 最老帧上有很多特征观测约束
        */
        // 1. 上一次优化以后留下的 prior 里可能存在约束   // 先验残差
        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set; //边缘化的优化变量的位置_drop_set
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            // last_marginalization_parameter_blocks 是上一轮留下来的残差块
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0]) // 最老的一帧给丢掉 需要marg掉的优化变量，也就是滑窗内第一个变量,para_Pose[0]和para_SpeedBias[0]
                    drop_set.push_back(i);
            }
            // 创建新的marg因子 construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

            /* 是为了将不同的损失函数_cost_function以及优化变量_parameter_blocks统一起来再一起添加到marginalization_info中
            ResidualBlockInfo(ceres::CostFunction *_cost_function,
                            ceres::LossFunction *_loss_function,
                            std::vector<double *> _parameter_blocks,
                            std::vector<int> _drop_set) */
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set); //这一步添加了marg信息

            // 将上一步marginalization后的信息作为先验信息
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // 添加IMU的marg信息
        // 然后添加第0帧和第1帧之间的IMU预积分值以及第0帧和第1帧相关优化变量
        if (USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1}); // vector<int>{0, 1} 表示要marg的参数下标，比如这里对应para_Pose[0], para_SpeedBias[0]  //这里是0,1的原因是0和1是para_Pose[0], para_SpeedBias[0]是需要marg的变量
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // 3. 最老帧上有很多特征观测约束  // 添加视觉的maeg信息
        {
            int feature_index = -1;
            //这里是遍历滑窗所有的特征点
            for (auto &it_per_id : f_manager.feature) // 遍历所有特征
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; //这里是从特征点的第一个观察帧开始
                if (imu_i != 0)                                       // 如果这个特征的初始帧 不对应 要marg掉的最老帧0, 那就不用marg这个特征。即marg掉帧的时候，我们marg掉这帧上三角化的那些点
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                for (auto &it_per_frame : it_per_id.feature_per_frame) //遍历这个特征的所有观测
                {
                    imu_j++;
                    if (imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]}, //优化变量
                                                                                       vector<int>{0, 3});                                                                                             //为0和3的原因是，para_Pose[imu_i]是第一帧的位姿，需要marg掉，而3是para_Feature[feature_index]是和第一帧相关的特征点，需要marg掉
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if (STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if (imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]}, //优化变量
                                                                                           vector<int>{0, 4});                                                                                                              //为0和4的原因是，para_Pose[imu_i]是第一帧的位姿，需要marg掉，而4是para_Feature[feature_index]是和第一帧相关的特征点，需要marg掉
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2}); //为2的原因是para_Feature[feature_index]是和第一帧相关的特征点，需要marg掉
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }
        if (NUM_OF_LINE > 0)
        {
            // Line feature
            int linefeature_index = -1;
            for (auto &it_per_id : f_manager.linefeature)
            {
                it_per_id.used_num = it_per_id.linefeature_per_frame.size();                                                        // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
                if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation)) // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
                    continue;
                ++linefeature_index; // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0) // 如果这个特征的初始帧 不对应 要marg掉的最老帧0, 那就不用marg这个特征。即marg掉帧的时候，我们marg掉这帧上三角化的那些点
                    continue;

                Vector4d pts_i = it_per_id.linefeature_per_frame[0].lineobs;
                for (auto &it_per_frame : it_per_id.linefeature_per_frame)
                {

                    //                     imu_j++;

                    //                     std::vector<int> drop_set;
                    //                     if(imu_i == imu_j)
                    //                     {
                    // //                        drop_set = vector<int>{0, 2};   // marg pose and feature,  !!!! do not need marg, just drop they  !!!
                    //                         continue;
                    //                     }else
                    //                     {
                    //                         drop_set = vector<int>{2};      // marg feature
                    //                     }

                    //                     Vector4d obs = it_per_frame.lineobs;                          // 在第j帧图像上的观测
                    //                     lineProjectionFactor *f = new lineProjectionFactor(obs);     // 特征重投影误差

                    //                     ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                    //                                                                                    vector<double *>{para_Pose[imu_j], para_Ex_Pose[0], para_LineFeature[linefeature_index]},
                    //                                                                                    drop_set);// vector<int>{0, 2} 表示要marg的参数下标，比如这里对应para_Pose[imu_i], para_Feature[feature_index]
                    //                     marginalization_info->addResidualBlockInfo(residual_block_info);
                    imu_j++;

                    if (imu_i != imu_j)
                    {
                        Vector4d pts_j = it_per_frame.lineobs;                     // 在第j帧图像上的观测
                        lineProjectionFactor *f = new lineProjectionFactor(pts_j); // 特征重投影误差
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_j], para_Ex_Pose[0], para_LineFeature[linefeature_index]},
                                                                                       vector<int>{2}); // 2表示要marg的参数下标，比如这里对应 para_Feature[feature_index]
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if (STEREO && it_per_frame.is_stereo && NUM_OF_LINE == 2)
                    {
                        if (imu_i != imu_j)
                        {
                            Vector4d pts_j_right = it_per_frame.lineobs_R;
                            lineProjectionFactor *f = new lineProjectionFactor(pts_j_right); // 特征重投影误差
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_j], para_Ex_Pose[1], para_LineFeature[linefeature_index]},
                                                                                           vector<int>{2}); // 2表示要marg的参数下标，比如这里对应 para_Feature[feature_index]
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                    if (imu_i == imu_j) // marg pose and feature,  !!!! do not need marg, just drop they  !!!
                    {
                        continue;
                    }

                    // if(STEREO && it_per_frame.is_stereo)
                    // {
                    //     Vector3d pts_j_right = it_per_frame.lineobs_R;
                    //     if(imu_i != imu_j)
                    //     {
                    //         lineProjectionFactor *f = new lineProjectionFactor(pts_j_right);     // 特征重投影误差
                    //         ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                    //             vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},//优化变量
                    //             vector<int>{0, 4});//为0和3的原因是，para_Pose[imu_i]是第一帧的位姿，需要marg掉，而3是para_Feature[feature_index]是和第一帧相关的特征点，需要marg掉
                    //         marginalization_info->addResidualBlockInfo(residual_block_info);
                    //     }
                    //     else
                    //     {
                    //         lineProjectionFactor *f = new lineProjectionFactor(pts_j_right);     // 特征重投影误差
                    //         ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                    //                                                                        vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                    //                                                                        vector<int>{2});
                    //         marginalization_info->addResidualBlockInfo(residual_block_info);
                    //     }
                    // }

                    //                     if(imu_i == imu_j)
                    //                     {
                    // //                        drop_set = vector<int>{0, 2};   // marg pose and feature,  !!!! do not need marg, just drop they  !!!
                    //                         continue;
                    //                     }else
                    //                     {
                    //                         drop_set = vector<int>{2};      // marg feature
                    //                     }

                    //                     Vector4d obs = it_per_frame.lineobs;                          // 在第j帧图像上的观测
                    //                     lineProjectionFactor *f = new lineProjectionFactor(obs);     // 特征重投影误差

                    //                     ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                    //                                                                                    vector<double *>{para_Pose[imu_j], para_Ex_Pose[0], para_LineFeature[linefeature_index]},
                    //                                                                                    drop_set);// vector<int>{0, 2} 表示要marg的参数下标，比如这里对应para_Pose[imu_i], para_Feature[feature_index]
                    //                     marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }

        TicToc t_pre_margin;
        // 上面通过调用 addResidualBlockInfo() 已经确定优化变量的数量、存储位置、长度以及待优化变量的数量以及存储位置，
        //-------------------------- 下面就需要调用 preMarginalize() 进行预处理
        marginalization_info->preMarginalize(); //执行Evaluate()函数
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        //------------------------调用 marginalize 正式开始边缘化
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        //------------------------在optimization的最后会有一部滑窗预移动的操作
        // 值得注意的是，这里仅仅是相当于将指针进行了一次移动，指针对应的数据还是旧数据，因此需要结合后面调用的 slideWindow() 函数才能实现真正的滑窗移动
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++) //从1开始，因为第一帧的状态不要了
        {
            //这一步的操作指的是第i的位置存放的的是i-1的内容，这就意味着窗口向前移动了一格
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1]; //因此para_Pose这些变量都是双指针变量，因此这一步是指针操作
            if (USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;                     //删除掉上一次的marg相关的内容
        last_marginalization_info = marginalization_info;         // marg相关内容的递归
        last_marginalization_parameter_blocks = parameter_blocks; //优化变量的递归，这里面仅仅是指针
    }
    else //边缘化掉次新帧
    {
        // cout<<"边缘化掉次新帧"<<endl;
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    sum_marg_time_ += t_whole_marginalization.toc();
    mean_marg_time_ = sum_marg_time_ / frame_cnt_;
    ROS_DEBUG("whole marginalization costs: %f", mean_marg_time_);

    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

// 基于滑动窗口的紧耦合的非线性优化，残差项的构造和求解

// 滑动窗口法
void Estimator::slideWindow()
{
    ROS_DEBUG("start slideWindow");
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    // 道理很简单,就是把前后元素交换,这样的话最后的结果是1234567890
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]); //交换
                Ps[i].swap(Ps[i + 1]);
                if (USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]); //交换预积分值

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }

            // 下边这一步的结果应该是1234567899
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if (USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];                                                                  //将预积分的最后一个值删除
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]}; //在构造一个新的

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0); //找到第一个
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }

    // marg掉倒数第二帧,也很简单,另倒数第二个等于新的一个就可以
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if (USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

// 滑到倒数第二帧,作用主要是删除特征点
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

// 滑掉最老的那一帧,,作用主要是删除特征点
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false; //判断是否处于初始化
    if (shift_depth)                                             //如果不是初始化
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

// 得到当前帧的变换矩阵T
void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

// 得到某一个index处图像的变换矩阵T
void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

// 在下一帧上预测这些特征点
void Estimator::predictPtsInNextFrame()
{
    // printf("predict pts in next frame\n");
    if (frame_count < 2)
        return;

    // TODO:动态检测 这里做了一个简单的预测!!可以用来做动态检测
    // 使用匀速模型预测下一个位置predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;

    // 获得当前帧和上一阵的位姿
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);

    // 预测下一帧的位姿
    nextT = curT * (prevT.inverse() * curT); //假设这一次的位姿变化和上一次相同!
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature) //对于当前帧所有的特征点
    {
        if (it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;                                         // 第一个观测到该特征点的帧
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1; //最后一个观测到该特征点的帧,就是int endFrame();
            // printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if ((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    // printf("estimator output %d predict pts\n",(int)predictPts.size());
}

// 计算重投影误差
double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                    Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

// 移除野点,返回野点的迭代器
// 移除重投影误差大于3个像素的
void Estimator::outliersRejection(set<int> &removeIndex)
{
    // return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue; //跳出本次循环
        feature_index++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                     Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                     depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if (STEREO && it_per_frame.is_stereo)
            {

                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }
        double ave_err = err / errCnt;
        if (ave_err * FOCAL_LENGTH > 3) //误差大于三个像素
            removeIndex.insert(it_per_id.feature_id);
    }
}
// 使用上一时刻的姿态进行快速的imu预积分
// 用来预测最新P,V,Q的姿态
// -latest_p,latest_q,latest_v,latest_acc_0,latest_gyr_0 最新时刻的姿态。这个的作用是为了刷新姿态的输出，但是这个值的误差相对会比较大，是未经过非线性优化获取的初始值。
void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

// 让此时刻的值都等于上一时刻的值,用来更新状态
void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while (!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
