#include "linefeature_tracker.h"
#include "../utility/tic_toc.h"
#include "../estimator/estimator.h"
#include "feature_tracker.h"
#include "../estimator/parameters.h"
#include<algorithm>
// #include "line_descriptor/src/precomp_custom.hpp"

LineFeatureTracker::LineFeatureTracker()
{
    allfeature_cnt = 0;//计算线特征的id
    frame_cnt = 0;//计算一共多少帧了
    sum_time = 0.0;
}


//把线端点的像素坐标根据内参转换为归一化坐标
vector<Line> LineFeatureTracker::undistortedLineEndPoints(cv::Mat K_,FrameLinesPtr frame_)
{
    vector<Line> un_lines;
    un_lines = frame_->vecLine;
    float fx = K_.at<float>(0, 0);
    float fy = K_.at<float>(1, 1);
    float cx = K_.at<float>(0, 2);
    float cy = K_.at<float>(1, 2);
    for (unsigned int i = 0; i <frame_->vecLine.size(); i++)
    {
        un_lines[i].StartPt.x = (frame_->vecLine[i].StartPt.x - cx)/fx;
        un_lines[i].StartPt.y = (frame_->vecLine[i].StartPt.y - cy)/fy;
        un_lines[i].EndPt.x = (frame_->vecLine[i].EndPt.x - cx)/fx;
        un_lines[i].EndPt.y = (frame_->vecLine[i].EndPt.y - cy)/fy;
    }
    return un_lines;
}


int frame_num = 0;
#define MATCHES_DIST_THRESHOLD 30
void visualize_line_match(Mat imageMat1, Mat imageMat2,
                          std::vector<KeyLine> octave0_1, std::vector<KeyLine>octave0_2,
                          std::vector<DMatch> good_matches)
{
    //	Mat img_1;
    cv::Mat img1,img2;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    if (imageMat2.channels() != 3){
        cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
    }
    else{
        img2 = imageMat2;
    }

    cv::Mat lsd_outImg;//线特征匹配输出的图像,将两个图像合并到了一起
    std::vector<char> lsd_mask( good_matches.size(), 1 );
    // drawLineMatches( img1, octave0_1, img2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ),Scalar::all( -1 ), lsd_mask,DrawLinesMatchesFlags::DEFAULT );

    drawLineMatches( img1, octave0_1, img2, octave0_2, good_matches, lsd_outImg, cv::Scalar(255, 0, 0),cv::Scalar(0, 0, 255), lsd_mask,DrawLinesMatchesFlags::DEFAULT );

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < good_matches.size(); ++k) {
        DMatch mt = good_matches[k];

        KeyLine line1 = octave0_1[mt.queryIdx];  // trainIdx
        KeyLine line2 = octave0_2[mt.trainIdx];  //queryIdx


        // unsigned int r = lowest + int(rand() % range);
        // unsigned int g = lowest + int(rand() % range);
        // unsigned int b = lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(line1.startPointX), int(line1.startPointY));
        cv::Point endPoint = cv::Point(int(line1.endPointX), int(line1.endPointY));
        // cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);
        cv::line(img1, startPoint, endPoint, cv::Scalar(255, 0, 0),2 ,8);

        cv::Point startPoint2 = cv::Point(int(line2.startPointX), int(line2.startPointY));
        cv::Point endPoint2 = cv::Point(int(line2.endPointX), int(line2.endPointY));
        // cv::line(img2, startPoint2, endPoint2, cv::Scalar(r, g, b),2, 8);
        cv::line(img2, startPoint2, endPoint2, cv::Scalar(0, 0, 255),2, 8);
        cv::line(img2, startPoint, startPoint2, cv::Scalar(0, 0, 0),4, 8);
        cv::line(img2, endPoint, endPoint2, cv::Scalar(0, 0, 0),4, 8);

    }
    /* plot matches */
    // cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);

    namedWindow("LSD matches", CV_WINDOW_NORMAL);
    imshow( "LSD matches", lsd_outImg );
    string name = to_string(frame_num);
    string path = "/home/zj/output/feature_tracker/image/";
    name = path + name + ".jpg";
    frame_num ++;
    imwrite(name, lsd_outImg);
    namedWindow("LSD matches1", CV_WINDOW_NORMAL);
    namedWindow("LSD matches2", CV_WINDOW_NORMAL);
    imshow("LSD matches1", img1);
    imshow("LSD matches2", img2);
    waitKey(1);
}


void visualize_line(Mat imageMat1,std::vector<KeyLine> octave0_1)
{
    //	Mat img_1;
    cv::Mat img1;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < octave0_1.size(); ++k) {

        unsigned int r = 255; //lowest + int(rand() % range);
        unsigned int g = 255; //lowest + int(rand() % range);
        unsigned int b = 0;  //lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(octave0_1[k].startPointX), int(octave0_1[k].startPointY));
        cv::Point endPoint = cv::Point(int(octave0_1[k].endPointX), int(octave0_1[k].endPointY));
        cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);//黄色
        // cv::circle(img1, startPoint, 2, cv::Scalar(255, 0, 0), 5);
        // cv::circle(img1, endPoint, 2, cv::Scalar(0, 255, 0), 5);


    }
    /* plot matches */
    /*
    cv::Mat lsd_outImg;
    std::vector<char> lsd_mask( lsd_matches.size(), 1 );
    drawLineMatches( imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ), Scalar::all( -1 ), lsd_mask,
    DrawLinesMatchesFlags::DEFAULT );

    imshow( "LSD matches", lsd_outImg );
    */
    namedWindow("LSD_C", CV_WINDOW_NORMAL);
    imshow("LSD_C", img1);
    waitKey(1);
}


//从图像中进行线特征的提取、跟踪和补充
cv::Mat last_unsuccess_image;
vector< KeyLine > last_unsuccess_keylsd;
vector< int >  last_unsuccess_id;
Mat last_unsuccess_lbd_descr;
void LineFeatureTracker::readImage(map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>>* lineFeatureFrame, drawData* data,FeatureTracker track, double _cur_time, const cv::Mat _img, const cv::Mat _img1)
{
    // double frame_cnt = 0;
    // double sum_time = 0.0;
    // double mean_time = 0.0;
    cv::Mat img,img1;//左目和右目
    TicToc t_p;//计算重映射的时间，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程,用来进行去畸变
    frame_cnt++;
    //重映射，去畸变
    cv::remap(_img, img, track.undist_map1_[0], track.undist_map2_[0], CV_INTER_LINEAR);
    
    if(STEREO==1&&NUM_OF_LINE==2)
    {
        cv::remap(_img1, img1, track.undist_map1_[1], track.undist_map2_[1], CV_INTER_LINEAR);
    }

    // //重映射，去畸变
    //     img=_img;
    // if(STEREO==1)
    // {
    //     img1=_img1;
    // }
    // cv::imshow("_img",_img);
    // cv::imshow("img",img);
    // cv::waitKey(30);
    
    ROS_DEBUG("undistortImage costs: %fms", t_p.toc());
    // if (EQUALIZE)   // 直方图均衡化 ，前面已经进行过了
    // if (1)   // 直方图均衡化
    // {
    //     cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    //     clahe->apply(img, img);
    // }
    
    bool first_img = false;
    if (curframe_ == nullptr) // 系统初始化的第一帧图像
    {
        curframe_.reset(new FrameLines);
        prevframe_.reset(new FrameLines);
        curframe_->img = img;
        prevframe_->img = img;
        first_img = true;
        if(STEREO==1&&NUM_OF_LINE==2)
       {
        right_curframe_.reset(new FrameLines);
        right_curframe_->img = img1;
       }
    }
    else
    {   //当前帧
        curframe_.reset(new FrameLines);  // 初始化一个新的帧
        curframe_->img = img;
        if(STEREO==1&&NUM_OF_LINE==2)
       {
        right_curframe_.reset(new FrameLines);
        right_curframe_->img = img1;
       }
    }
    TicToc t_li;
    //生成一个LSD的线段检测器lsd_， 并设置相应的参数opt 
    Ptr<line_descriptor::LSDDetectorC> lsd_ = line_descriptor::LSDDetectorC::createLSDDetectorC();
    // lsd parameters
    line_descriptor::LSDDetectorC::LSDOptions opts;
    opts.refine       = 1;     //1     	The way found lines will be refined
    opts.scale        = 0.5;   //0.8   	The scale of the image that will be used to find the lines. Range (0..1].
    opts.sigma_scale  = 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
    opts.quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
    opts.ang_th       = 22.5;	//22.5	Gradient angle tolerance in degrees
    opts.log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
    opts.density_th   = 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
    opts.n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.
    double min_line_length = 0.125;  // Line segments shorter than that are rejected
    // opts.refine       = 1;
    // opts.scale        = 0.5;
    // opts.sigma_scale  = 0.6;
    // opts.quant        = 2.0;
    // opts.ang_th       = 22.5;
    // opts.log_eps      = 1.0;
    // opts.density_th   = 0.6;
    // opts.n_bins       = 1024;
    // double min_line_length = 0.125;
    opts.min_length   = min_line_length*(std::min(img.cols,img.rows));
    
    //构造存储左目线特征的容器 lsd 和关键线特征的容器 keylsd。
    std::vector<KeyLine> lsd, keylsd;
    //构造存储右目线特征的容器 lsd 和关键线特征的容器 keylsd。
    std::vector<KeyLine> right_lsd, right_keylsd;
	//void LSDDetectorC::detect( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, int scale, int numOctaves, const std::vector<Mat>& masks ) const
    lsd_->detect( img, lsd, 2, 1, opts);
    //visualize_line(img,lsd);
    // step 1: line extraction
    // TicToc t_li;
    // std::vector<KeyLine> lsd, keylsd;
    // Ptr<LSDDetector> lsd_;
    // lsd_ = cv::line_descriptor::LSDDetector::createLSDDetector();
    // lsd_->detect( img, lsd, 2, 2 );

    sum_time += t_li.toc();
     //若为双目
    if(STEREO==1&&NUM_OF_LINE==2)
    {
	//void LSDDetectorC::detect( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, int scale, int numOctaves, const std::vector<Mat>& masks ) const
    lsd_->detect( img1, right_lsd, 2, 1, opts);
    //visualize_line(img1,right_lsd);
    
    // cout<<"right_lsd.size(): "<<right_lsd.size()<<endl;
    sum_time += t_li.toc();
    }
    ROS_DEBUG("line detect costs: %fms, size: %d", t_li.toc(), lsd.size());

    //进一步计算左目线特征对应描述子，并根据octave和线段长度生成关键线特征的集合，同时添加描述子
    Mat lbd_descr, keylbd_descr;
    //进一步计算右目线特征对应描述子，并根据octave和线段长度生成关键线特征的集合，同时添加描述子
    Mat right_lbd_descr, right_keylbd_descr;
    // step 2: lbd descriptor
    TicToc t_lbd;
    Ptr<BinaryDescriptor> bd_ = BinaryDescriptor::createBinaryDescriptor(  );
    

    bd_->compute( img, lsd, lbd_descr );
    // std::cout<<"lbd_descr = "<<lbd_descr.size()<<std::endl;
//////////////////////////
    for ( int i = 0; i < (int) lsd.size(); i++ )//激活的线，且长度大于60
    {
        if( lsd[i].octave == 0 && lsd[i].lineLength >= 60)
        {
            keylsd.push_back( lsd[i] );
            keylbd_descr.push_back( lbd_descr.row( i ) );
        }
    }
    if(keylsd.size()==0)
    {
        ROS_WARN("keylsd is empty!!!");
        return;
    }
    
    
    // std::cout<<"lbd_descr = "<<lbd_descr.size()<<std::endl;
//    ROS_INFO("lbd_descr detect costs: %fms", keylsd.size() * t_lbd.toc() / lsd.size() );
    sum_time += keylsd.size() * t_lbd.toc() / lsd.size();

    if(STEREO==1&&NUM_OF_LINE==2)
    {
    Ptr<BinaryDescriptor> right_bd_ = BinaryDescriptor::createBinaryDescriptor(  );
    

    right_bd_->compute( img1, right_lsd, right_lbd_descr );
    // std::cout<<"lbd_descr = "<<lbd_descr.size()<<std::endl;
//////////////////////////
    for ( int i = 0; i < (int) right_lsd.size(); i++ )//激活的线，且长度大于60
    {
        if( right_lsd[i].octave == 0 && right_lsd[i].lineLength >= 60)
        {
            right_keylsd.push_back( right_lsd[i] );
            right_keylbd_descr.push_back( right_lbd_descr.row( i ) );
        }
    }
    // visualize_line(img1,right_keylsd);
    // cout<<"right_keylsd: "<<right_keylsd.size()<<endl;
    // cout<<"right_keylbd_descr = "<<lbd_descr.size()<<endl;
//    ROS_INFO("lbd_descr detect costs: %fms", keylsd.size() * t_lbd.toc() / lsd.size() );
    sum_time += right_keylsd.size() * t_lbd.toc() / right_lsd.size();
    }
///////////////
    

    //赋值给当前帧curframe， 若为第一帧则为每条线特征赋予id, 否则先给一个-1的id
    curframe_->keylsd = keylsd;
    curframe_->lbd_descr = keylbd_descr;

    
    for (size_t i = 0; i < curframe_->keylsd.size(); ++i) {
        if(first_img)
            curframe_->lineID.push_back(allfeature_cnt++);
        else
            curframe_->lineID.push_back(-1);   // give a negative id
    }
    
    
    //若为双目
    if(STEREO==1&&NUM_OF_LINE==2)
    {
    
    //赋值给当前帧rigth_curframe， 全部给一个-1的id
    right_curframe_->keylsd = right_keylsd;
    right_curframe_->lbd_descr = right_keylbd_descr;
    for (size_t i = 0; i < right_curframe_->keylsd.size(); ++i)
        right_curframe_->lineID.push_back(-1);
    }

     if(SHOW_TRACK)
     {
         data->line_ids=curframe_->lineID;
         data->keylsd=keylsd;
         data->right_keylsd=right_keylsd;
     }

    if(1)//单目
    {
        //若前一帧的关键线特征个数大于0（不是第一帧）,则开始进行线特征跟踪和匹配。
        if(prevframe_->keylsd.size() > 0)
        {
            /* compute matches 计算匹配子 */
            // TicToc t_match;
            std::vector<DMatch> lsd_matches;
            Ptr<BinaryDescriptorMatcher> bdm_;
            bdm_ = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
            bdm_->match(curframe_->lbd_descr, prevframe_->lbd_descr, lsd_matches);//前面默认为新的图片，后面为旧的图片
    //        ROS_INFO("lbd_macht costs: %fms", t_match.toc());
            // sum_time += t_match.toc();
            // mean_time = sum_time/frame_cnt;
            // ROS_INFO("line feature tracker mean costs: %fms", mean_time);
            
            // cout<<"左目 lsd_matches: "<<lsd_matches.size()<<endl;
            


            /* select best matches 选择好的匹配*/
            std::vector<DMatch> good_matches;
            std::vector<KeyLine> good_Keylines;
            good_matches.clear();
            // cout<<"左目: lsd_matches[i].distance: ";
            for ( int i = 0; i < (int) lsd_matches.size(); i++ )
            {   
                // cout<<lsd_matches[i].distance<<" ";
                if( lsd_matches[i].distance < 30 ){//distance 表示的是特征点的相似程度

                    DMatch mt = lsd_matches[i];
                    KeyLine line1 =  curframe_->keylsd[mt.queryIdx] ;//当前要寻找匹配结果的点在它所在图片上的索引
                    KeyLine line2 =  prevframe_->keylsd[mt.trainIdx] ;//查找到的结果的索引
                    Point2f serr = line1.getStartPoint() - line2.getEndPoint();
                    Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                    // std::cout<<"11111111111111111 = "<<abs(line1.angle-line2.angle)<<std::endl;
                    if((serr.dot(serr) < 200 * 200) && (eerr.dot(eerr) < 200 * 200)&&abs(line1.angle-line2.angle)<0.1)   // 线段在图像里不会跑得特别远
                        good_matches.push_back( lsd_matches[i] );
                }
            }
            // int before=good_matches.size();
            // cout<<"左目处理前 good_matches: "<<before<<endl;
            // for(auto a: good_matches)
            // {
            //     cout<<a.queryIdx<<" "<<a.trainIdx<<endl;
                
            // }
            
            //按照匹配的线特征id排序，如果出现两个特征匹配到了同一个特征，distance小的优先
            sort(good_matches.begin(), good_matches.end(), [](const DMatch &a, const DMatch &b)
         {
            if(a.trainIdx<b.trainIdx)
            return true;
            else if(a.trainIdx==b.trainIdx)
            return a.distance<b.distance;
            else
            return false;
         });
        //去除重复的元素
        int j=0;
        for(int i=1;i<good_matches.size();i++)
        {
            if(good_matches[i].trainIdx!=good_matches[j].trainIdx)
                good_matches[++j]=good_matches[i];
        }
        good_matches.resize(j+1);

        // auto ite = unique(good_matches.begin(), good_matches.end());
        // //删除重复的元素
        // good_matches.erase(ite, good_matches.end());
            
            // int after=good_matches.size();
            // cout<<endl;
            // cout<<"左目处理后 good_matches: "<<after<<endl;
            // if(before!=after)
            // {
            //     cout<<"zjjjjj"<<endl;
            //     getchar();
            // }
            // for(auto a: good_matches)
            // {
            //     cout<<a.queryIdx<<" "<<a.trainIdx<<endl;
                
            // }

            //在success中保存对应的ID
            vector< int > success_id;
            // std::cout << curframe_->lineID.size() <<" " <<prevframe_->lineID.size();
            for (int k = 0; k < good_matches.size(); ++k) {
                DMatch mt = good_matches[k];
                curframe_->lineID[mt.queryIdx] = prevframe_->lineID[mt.trainIdx];
                success_id.push_back(prevframe_->lineID[mt.trainIdx]);
            }
            // cout<<"match after:"<<endl;
            // for(int i=0;i<curframe_->lineID.size();i++)
            // {
            //     cout<<i<<" "<<curframe_->lineID[i]<<endl;
            // }
            // cout<<endl;
            // cout<<endl;
            
            // cout<<"左目 success_id: "<<success_id.size()<<endl;
            // for(auto a: success_id)
            // {
            //     cout<<a<<" ";
            // }
            // cout<<endl;
            // visualize_line_match(curframe_->img.clone(), prevframe_->img.clone(), curframe_->keylsd, prevframe_->keylsd, good_matches);

            
            //将左目跟踪到的线和未跟踪到的线（新线）分别加入到vecLine_tracked 和 vecLine_new中，右目加入到right_vecLine_tracked 和 right_vecLine_new
            vector<KeyLine> vecLine_tracked, vecLine_new;
            vector< int > lineID_tracked, lineID_new;
            Mat DEscr_tracked, Descr_new;
            
            // 将左目跟踪的线和没跟踪上的线进行区分
            for (size_t i = 0; i < curframe_->keylsd.size(); ++i)
            {
                
                if( curframe_->lineID[i] == -1)//若没跟踪到
                {
                    curframe_->lineID[i] = allfeature_cnt++;
                    vecLine_new.push_back(curframe_->keylsd[i]);
                    lineID_new.push_back(curframe_->lineID[i]);
                    Descr_new.push_back( curframe_->lbd_descr.row( i ) );
                }
                
                else //若跟踪到
                {
                    vecLine_tracked.push_back(curframe_->keylsd[i]);
                    lineID_tracked.push_back(curframe_->lineID[i]);
                    DEscr_tracked.push_back( curframe_->lbd_descr.row( i ) );
                }
            }
            
            // cout<<"fen after:"<<endl;
            // for(int i=0;i<curframe_->lineID.size();i++)
            // {
            //     cout<<i<<" "<<curframe_->lineID[i]<<endl;
            // }
            // cout<<endl;
            // cout<<endl;

            /* 通过一个简单的线段角度判断（相对于图像x轴的斜率）来识别未跟踪到的线（新线）为水平线
            （horizontal line, [ π , 3 π / 4 ] [\pi, 3\pi/4][π,3π/4] or [ − 3 π / 4 , − π / 4 ] [-3\pi/4, -\pi/4][−3π/4,−π/4]）
            还是竖直线（vertical line, 其他角度）*/
            vector<KeyLine> h_Line_new, v_Line_new;
            vector< int > h_lineID_new,v_lineID_new;
            Mat h_Descr_new,v_Descr_new;
            for (size_t i = 0; i < vecLine_new.size(); ++i)
            {
                if((((vecLine_new[i].angle >= 3.14/4 && vecLine_new[i].angle <= 3*3.14/4))||(vecLine_new[i].angle <= -3.14/4 && vecLine_new[i].angle >= -3*3.14/4)))
                {
                    h_Line_new.push_back(vecLine_new[i]);
                    h_lineID_new.push_back(lineID_new[i]);
                    h_Descr_new.push_back(Descr_new.row( i ));
                }
                else
                {
                    v_Line_new.push_back(vecLine_new[i]);
                    v_lineID_new.push_back(lineID_new[i]);
                    v_Descr_new.push_back(Descr_new.row( i ));
                }      
            }
           
            /*把已跟踪的线进行水平线和竖直线的分类，进一步判断是否满足所要求的数量，不满足的话则补充新线。
            最后，更新当前帧的 关键线特征及其描述子和ID*/
            int h_line,v_line;
            h_line = v_line =0;
            for (size_t i = 0; i < vecLine_tracked.size(); ++i)
            {
                if((((vecLine_tracked[i].angle >= 3.14/4 && vecLine_tracked[i].angle <= 3*3.14/4))||(vecLine_tracked[i].angle <= -3.14/4 && vecLine_tracked[i].angle >= -3*3.14/4)))
                {
                    h_line ++;
                }
                else
                {
                    v_line ++;
                }
            }
            int diff_h = 35 - h_line;
            int diff_v = 35 - v_line;

            // std::cout<<"h_line = "<<h_line<<" v_line = "<<v_line<<std::endl;
            if( diff_h > 0)    // 补充线条
            {
                int kkk = 1;
                if(diff_h > h_Line_new.size())
                    diff_h = h_Line_new.size();
                else 
                    kkk = int(h_Line_new.size()/diff_h);
                for (int k = 0; k < diff_h; ++k) 
                {
                    vecLine_tracked.push_back(h_Line_new[k]);
                    lineID_tracked.push_back(h_lineID_new[k]);
                    DEscr_tracked.push_back(h_Descr_new.row(k));
                }
                // std::cout  <<"h_kkk = " <<kkk<<" diff_h = "<<diff_h<<" h_Line_new.size() = "<<h_Line_new.size()<<std::endl;
            }
            
            if( diff_v > 0)    // 补充线条
            {
                int kkk = 1;
                if(diff_v > v_Line_new.size())
                    diff_v = v_Line_new.size();
                else 
                    kkk = int(v_Line_new.size()/diff_v);
                for (int k = 0; k < diff_v; ++k)  
                {
                    vecLine_tracked.push_back(v_Line_new[k]);
                    lineID_tracked.push_back(v_lineID_new[k]);
                    DEscr_tracked.push_back(v_Descr_new.row(k));
                }            // std::cout  <<"v_kkk = " <<kkk<<" diff_v = "<<diff_v<<" v_Line_new.size() = "<<v_Line_new.size()<<std::endl;
            }


            // int diff_n = 50 - vecLine_tracked.size();  // 跟踪的线特征少于50了，那就补充新的线特征, 还差多少条线
            // if( diff_n > 0)    // 补充线条
            // {
            //     for (int k = 0; k < vecLine_new.size(); ++k) {
            //         vecLine_tracked.push_back(vecLine_new[k]);
            //         lineID_tracked.push_back(lineID_new[k]);
            //         DEscr_tracked.push_back(Descr_new.row(k));
            //     }
            // }
            
            
            curframe_->keylsd = vecLine_tracked;
            curframe_->lineID = lineID_tracked;
            curframe_->lbd_descr = DEscr_tracked;
    
        }
       
        // 将opencv的KeyLine线特征转化为本框架下使用的Line线特征，只使用了端点坐标和线段长度
            for (int j = 0; j < curframe_->keylsd.size(); ++j) {
                Line l;
                KeyLine lsd = curframe_->keylsd[j];
                l.StartPt = lsd.getStartPoint();
                l.EndPt = lsd.getEndPoint();
                l.length = lsd.lineLength;
                curframe_->vecLine.push_back(l);
            }
            //若是第一帧，则不进行任何操作并赋当前帧给前一帧， continue
            prevframe_ = curframe_;
        
    }


    if(STEREO==1&&NUM_OF_LINE==2)
    {
        
        if(1)//左目肯定有线特征的，和左目开始进行线特征跟踪和匹配。
        {
             /* compute matches 计算匹配子 */
            TicToc t_match;
            std::vector<DMatch> lsd_matches;
            Ptr<BinaryDescriptorMatcher> bdm_;
            bdm_ = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
            bdm_->match(right_curframe_->lbd_descr,curframe_->lbd_descr,lsd_matches);
            // cout<<"右目 lsd_matches: "<<lsd_matches.size()<<endl;
            sum_time += t_match.toc();
            mean_time = sum_time/frame_cnt;
            // ROS_INFO("line feature tracker mean costs: %fms", mean_time);
            /* select best matches 选择好的匹配*/
            std::vector<DMatch> good_matches;
            std::vector<KeyLine> good_Keylines;
            good_matches.clear();
            // cout<<"右目: lsd_matches[i].distance: ";
            for ( int i = 0; i < (int) lsd_matches.size(); i++ )
            {
                // cout<<lsd_matches[i].distance<<" ";
                if( lsd_matches[i].distance < 30 ){//distance 表示的是特征点的相似程度

                    DMatch mt = lsd_matches[i];
                    KeyLine line1 =  right_curframe_->keylsd[mt.queryIdx] ;
                    KeyLine line2 =  curframe_->keylsd[mt.trainIdx];
                    Point2f serr = line1.getStartPoint() - line2.getEndPoint();
                    Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                    // std::cout<<"11111111111111111 = "<<abs(line1.angle-line2.angle)<<std::endl;
                    if((serr.dot(serr) < 200 * 200) && (eerr.dot(eerr) < 200 * 200)&&abs(line1.angle-line2.angle)<0.1)   // 线段在图像里不会跑得特别远
                        good_matches.push_back( lsd_matches[i] );
                }
            }
            // cout<<endl;
            // cout<<"右目 good_matches: "<<good_matches.size()<<endl;
            // right_curframe_->lineID=vector<int>(good_matches.size());
            //在success中保存对应的ID
            // 只保留了和左目匹配上的
            vector< int >right_success_id;
            // std::cout << curframe_->lineID.size() <<" " <<prevframe_->lineID.size();
            for (int k = 0; k < good_matches.size(); ++k) {
                DMatch mt = good_matches[k];
                right_curframe_->lineID[mt.queryIdx]=curframe_->lineID[mt.trainIdx];
                right_success_id.push_back(curframe_->lineID[mt.trainIdx]);
            }
            // cout<<"右目 right_success_id: "<<right_success_id.size()<<endl;
            // for(int i=0;i<right_success_id.size();i++)
            // cout<<right_success_id[i]<<" ";
            // cout<<endl;
            
            

            // visualize_line_match(curframe_->img.clone(), right_curframe_->img.clone(), curframe_->keylsd, right_curframe_->keylsd, good_matches);

            
            //将左目跟踪到的线和未跟踪到的线（新线）分别加入到vecLine_tracked 和 vecLine_new中，右目加入到right_vecLine_tracked 和 right_vecLine_new
            vector<KeyLine> right_vecLine_tracked,right_vecLine_new;
            vector< int > right_lineID_tracked, right_lineID_new;
            Mat right_DEscr_tracked, right_Descr_new;
            
            // 将右目跟踪的线和没跟踪上的线进行区分
            for (size_t i = 0; i < right_curframe_->keylsd.size(); ++i)
            {
                if( right_curframe_->lineID[i]!=-1)//若跟踪到
                {
                    right_vecLine_tracked.push_back(right_curframe_->keylsd[i]);
                    right_lineID_tracked.push_back(right_curframe_->lineID[i]);
                    right_DEscr_tracked.push_back( right_curframe_->lbd_descr.row( i ) );
                }
            }

            //右目
            right_curframe_->lineID = right_lineID_tracked;
            right_curframe_->keylsd=right_vecLine_tracked;
            right_curframe_->lbd_descr = right_DEscr_tracked;
            
            if(SHOW_TRACK)
            {
                data->match_ids=right_lineID_tracked;
                data->match_keylsd=right_vecLine_tracked;
            }

     

            
         }
        
        
             // 右目将opencv的KeyLine线特征转化为本框架下使用的Line线特征，只使用了端点坐标和线段长度
            for (int j = 0; j < right_curframe_->keylsd.size(); ++j) {
                Line l;
                KeyLine lsd = right_curframe_->keylsd[j];
                l.StartPt = lsd.getStartPoint();
                l.EndPt = lsd.getEndPoint();
                l.length = lsd.lineLength;
                right_curframe_->vecLine.push_back(l);
            }
    }
    

    
    
   // ----------------------构建线特征featureFrame，并加入线特征
    vector<Line> un_lines =undistortedLineEndPoints(track.K_[0],curframe_);
    auto &ids =curframe_->lineID;
    // cout<<"左目线特征的id： 共"<<ids.size()<<endl;
    // for(int i: ids)
    //     cout<<i<<" ";
    // cout<<endl;
    
    for(size_t i=0;i<ids.size();i++)
    {
        int line_id = ids[i];//该帧上的线特征编号
        double start_x,start_y;//起点归一化平面坐标
        start_x = un_lines[i].StartPt.x;
        start_y = un_lines[i].StartPt.y;
        double end_x,end_y;//终点归一化平面坐标
        end_x=un_lines[i].EndPt.x;
        end_y=un_lines[i].EndPt.y;
        int camera_id=0;//相机的id

        Eigen::Matrix<double, 4, 1> start_end_xy;
        start_end_xy <<start_x,start_y,end_x,end_y;
        (*lineFeatureFrame)[line_id].emplace_back(camera_id,start_end_xy);
        // 线特征的id，相机id（0或1） 和 start_end_xy（线特征起点空间坐标，终点空间坐标）
    }
    
    //右目
    if(STEREO==1&&NUM_OF_LINE==2)
    {
        // ----------------------构建线特征featureFrame，并加入线特征
        auto un_lines =undistortedLineEndPoints(track.K_[1],right_curframe_);
        auto &ids =right_curframe_->lineID;
        // cout<<"右目线特征的id: 共"<<ids.size()<<endl;
        // for(int i: ids)
        // cout<<i<<" ";
        // cout<<endl;
        for(size_t i=0;i<ids.size();i++)
        {
            int line_id = ids[i];//该帧上的线特征编号
            double start_x,start_y;//起点归一化平面坐标
            start_x = un_lines[i].StartPt.x;
            start_y = un_lines[i].StartPt.y;
            double end_x,end_y;//终点归一化平面坐标
            end_x=un_lines[i].EndPt.x;
            end_y=un_lines[i].EndPt.y;
            int camera_id=1;//相机的id

            Eigen::Matrix<double, 4, 1> start_end_xy;
            start_end_xy <<start_x,start_y,end_x,end_y;
            (*lineFeatureFrame)[line_id].emplace_back(camera_id,start_end_xy);
            // 线特征的id，相机id（0或1） 和 start_end_xy（线特征起点空间坐标，终点空间坐标）
        }
    }
    
    
    // cout<<"lineFeatureFrame:"<<lineFeatureFrame.size()<<endl;
    // for(auto a:lineFeatureFrame)
    // {
    //     cout<<"线特征id:"<<a.first<<" 被:"<<a.second.size()<<"个相机观测到"<<endl;
    // }
    
}
