
#pragma once

#include <iostream>
#include <queue>

#include "../../../camera_models/include/camodocal/camera_models/CameraFactory.h"
#include "../../../camera_models/include/camodocal/camera_models/CataCamera.h"
#include "../../../camera_models/include/camodocal/camera_models/PinholeCamera.h"
#include "../../../camera_models/include/camodocal/camera_models/EquidistantCamera.h"

#include "../estimator/parameters.h"

#include "../utility/tic_toc.h"

// #include <opencv2/line_descriptor.hpp>
#include <opencv2/features2d.hpp>

#include "line_descriptor/include/line_descriptor_custom.hpp"
#include "feature_tracker.h"

using namespace cv::line_descriptor;
using namespace std;
using namespace cv;
using namespace camodocal;



struct Line
{
	Point2f StartPt;
	Point2f EndPt;
	float lineWidth;
	Point2f Vp;

	Point2f Center;
	Point2f unitDir; // [cos(theta), sin(theta)]
	float length;
	float theta;

	// para_a * x + para_b * y + c = 0
	float para_a;
	float para_b;
	float para_c;

	float image_dx;
	float image_dy;
    float line_grad_avg;

	float xMin;
	float xMax;
	float yMin;
	float yMax;
	unsigned short id;
	int colorIdx;
};

class FrameLines
{
public:
    int frame_id;
    Mat img;
    
    vector<Line> vecLine;
    vector< int > lineID;

    // opencv3 lsd+lbd
    std::vector<KeyLine> keylsd;
    Mat lbd_descr;
};
typedef shared_ptr< FrameLines > FrameLinesPtr;


class LineFeatureTracker
{
  public:
    LineFeatureTracker();

    
    vector<Line> undistortedLineEndPoints(cv::Mat K_,FrameLinesPtr prevframe_);//把线端点的像素坐标根据内参转换为归一化坐标

    void readImage(map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>>* lineFeatureFrame, drawData* data, FeatureTracker track, double _cur_time, const cv::Mat _img, const cv::Mat _img1 = cv::Mat());
    // map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> readImage(FeatureTracker track,double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    FrameLinesPtr prevframe_, curframe_;//左目的前一帧和当前帧
    FrameLinesPtr right_curframe_;//右目的当前帧


    camodocal::CameraPtr m_camera;       // pinhole camera

    int frame_cnt;
    vector<int> ids;                     // 每个特征点的id
    vector<int> linetrack_cnt;           // 记录某个特征已经跟踪多少帧了，即被多少帧看到了
    int allfeature_cnt;                  // 用来统计左目整个地图中有了多少条线，它将用来赋值
    int right_allfeature_cnt;            //用来统计右目整个地图中有了多少条线，它将用来赋值
    double sum_time;
    double mean_time;
};
