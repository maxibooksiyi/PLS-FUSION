/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>
#include "../utility/line_geometry.h"
#include "parameters.h"
#include "../utility/tic_toc.h"



// 它指的是空间特征点P1映射到frame1或frame2上对应的图像坐标、特征点的跟踪速度、空间坐标等属性都封装到类FeaturePerFrame中
class FeaturePerFrame// 特征点在每一帧上的属性
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
        is_stereo = false;
        is_used=true;
    }

    // FIXME:双目
    void rightObservation(const Eigen::Matrix<double, 7, 1> &_point)
    {
        pointRight.x() = _point(0);
        pointRight.y() = _point(1);
        pointRight.z() = _point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocityRight.x() = _point(5); 
        velocityRight.y() = _point(6); 
        is_stereo = true;
        is_used=true;
    }
    double cur_td;
    Vector3d point, pointRight;
    Vector2d uv, uvRight;
    Vector2d velocity, velocityRight;
    bool is_stereo;

    //线特征添加的
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

// 就特征点P1来说，它被两个帧观测到，第一次观测到P1的帧为frame1,即start_frame=1，
// 最后一次观测到P1的帧为frame2,即endframe()=2,并把start_frame~endframe() 对应帧的属性存储起来，
class FeaturePerId// 管理一个特征点
{
  public:
    const int feature_id; //特征点id
    int start_frame;//第一次出现该特征点的帧号

    /*class FeaturePerFrame
    它指的是空间特征点P1映射到frame1或frame2上对应的图像坐标、特征点的跟踪速度、空间坐标等属性都封装到类FeaturePerFrame中*/
    //这个特征点在所有观测到他的图像上的性质
    vector<FeaturePerFrame> feature_per_frame; 

    int used_num;//出现的次数
    bool is_outlier;
    bool is_margin;
    double estimated_depth; //逆深度
    int solve_flag; // 该特征点的状态，是否被三角 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    // 构造函数
    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();//得到该特征点最后一次被跟踪到的帧号
};


//管理一个线特征所在的帧的属性
class lineFeaturePerFrame
{
public:
    lineFeaturePerFrame(const Vector4d &line)
    {
        lineobs = line;
        is_used=true;
    }
    

    // FIXME:双目
    void rightObservation(const Vector4d &line)
    {
        lineobs_R=line;
        is_stereo = true;
        is_used=true;
    }
    
    bool is_stereo;
    Vector4d lineobs;   // 每一帧上的观测
    Vector4d lineobs_R;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

//管理一个线特征
class lineFeaturePerId
{
public:
    const int feature_id;
    int start_frame;

    //  feature_per_frame 是个向量容器，存着这个特征在每一帧上的观测量。
    //                    如：feature_per_frame[0]，存的是ft在start_frame上的观测值; feature_per_frame[1]存的是start_frame+1上的观测
    vector<lineFeaturePerFrame> linefeature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    bool is_triangulation;
    Vector6d line_plucker;

    Vector4d obs_init;
    Vector4d obs_j;
    Vector6d line_plk_init; // used to debug
    Vector3d ptw1;  // used to debug
    Vector3d ptw2;  // used to debug
    Eigen::Vector3d tj_;   // tij
    Eigen::Matrix3d Rj_;
    Eigen::Vector3d ti_;   // tij
    Eigen::Matrix3d Ri_;
    int removed_cnt;
    int all_obs_cnt;    // 总共观测多少次了？

    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    lineFeaturePerId(int _feature_id, int _start_frame)
            : feature_id(_feature_id), start_frame(_start_frame),
              used_num(0), solve_flag(0),is_triangulation(false)
    {
        removed_cnt = 0;
        all_obs_cnt = 1;
    }

    int endFrame();
};

// 特征点的管理类
class FeatureManager//管理所有的特征点，具体关系可以看（在代码中分析VINS---图解特征点管理(feature_manager.h)）这个博客
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);
    void clearState();
    int getFeatureCount();

    int getLineFeatureCount();
    MatrixXd getLineOrthVector(Vector3d Ps[],Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    void setLineOrth(MatrixXd x, Vector3d Ps[], Matrix3d Rs[],Vector3d tic[], Matrix3d ric[]);
    MatrixXd getLineOrthVectorInCamera();
    void setLineOrthInCamera(MatrixXd x);
    double reprojection_error( Vector4d obs, Matrix3d Rwc, Vector3d twc, Vector6d line_w );
    void removeLineOutlier(Vector3d Ps[],Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    void removeLineOutlier();//移除野线

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &lines, double td);
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void debugShow();
    void removeFailures();
    void clearDepth();
    VectorXd getDepthVector();
    void triangulateLine(int frameCnt,Vector3d Ps[], Matrix3d Rs[],Vector3d tic[], Matrix3d ric[]);//单目线特征三角化
    void triangulateLine(double baseline);  // 双目线特征三角化
    void triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                            Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, 
                            vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier(set<int> &outlierIndex);
    

    /*class FeaturePerId
    管理一个特征点
    就特征点P1来说，它被两个帧观测到，第一次观测到P1的帧为frame1,即start_frame=1，
    最后一次观测到P1的帧为frame2,即endframe()=2,并把start_frame~endframe() 对应帧的属性存储起来，    */
    list<FeaturePerId> feature; // 里边放的是特征点,
    list<lineFeaturePerId> linefeature;//里面放的是线特征
    int last_track_num;
    double last_average_parallax;
    int new_feature_num;
    int new_linefeature_num;
    int long_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[2];
};

#endif