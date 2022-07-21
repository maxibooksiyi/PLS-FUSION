cmake_minimum_required(VERSION 2.8.3)
# 包名
project(pls_vins)

#设置变量
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# 这里的这个find_package的写法也是应该注意学习的
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)

# Set(OpenCV_DIR "/home/zj/CLionProjects/opencv3.4/build") # important find opencv version
#Set(OpenCV_DIR "/home/zj/CLionProjects/opencv3.4/build") # important find opencv version
find_package(OpenCV 3.2 QUIET)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)

#头文件
include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}
/data1/zhoujin/line_catkin_ws/src/PLS-VINS/vins_estimator/src/featureTracker/line_descriptor/include
)

find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)



catkin_package()
#生成库文件
add_library(pls_vins_lib
    src/estimator/parameters.cpp
    src/estimator/feature_manager.cpp
    src/estimator/estimator.cpp
    
  
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/projection_factor.cpp
    src/factor/marginalization_factor.cpp
    # src/factor/line_parameterization.h
    src/factor/line_parameterization.cpp
    # src/factor/line_projection_factor.h
    src/factor/line_projection_factor.cpp
    

    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/utility/line_geometry.cpp
    # src/utility/line_geometry.h

    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp

    
    src/featureTracker/feature_tracker.cpp
    src/featureTracker/linefeature_tracker.cpp
    src/featureTracker/line_descriptor/src/binary_descriptor_custom.cpp
    src/featureTracker/line_descriptor/src/binary_descriptor_matcher.cpp
    src/featureTracker/line_descriptor/src/bitarray_custom.hpp
    src/featureTracker/line_descriptor/src/bitops_custom.hpp
    src/featureTracker/line_descriptor/src/draw_custom.cpp
    src/featureTracker/line_descriptor/src/LSDDetector_custom.cpp
    src/featureTracker/line_descriptor/src/precomp_custom.hpp
    src/featureTracker/line_descriptor/src/types_custom.hpp

    src/sophus/common.hpp
    src/sophus/types.hpp
    src/sophus/so3.hpp

    
    )
#链接库文件
target_link_libraries(pls_vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}
/data1/zhoujin/line_catkin_ws/src/PLS-VINS/vins_estimator/src/featureTracker/line_descriptor/lib/liblinedesc.so  # line feature
)

#生成可执行文件
add_executable(pls_vins_node src/rosNodeTest.cpp)
target_link_libraries(pls_vins_node pls_vins_lib) 

add_executable(pls_kitti_odom_test src/KITTIOdomTest.cpp)
target_link_libraries(pls_kitti_odom_test pls_vins_lib) 

add_executable(pls_kitti_gps_test src/KITTIGPSTest.cpp)
target_link_libraries(pls_kitti_gps_test pls_vins_lib) 

