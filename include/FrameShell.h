#pragma once
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include "KeyFrame.h"

namespace SLAM
{

struct FrameShell
{
    void extractORBPoints();

    double timeStamp;
    unsigned long frameId;

    std::vector<cv::KeyPoint> keypoints;
    std::vector<float> pointDepth;
    std::vector<float> rightCoord;
    std::vector<bool> outliers;

    KeyFrame *parent;
};

} // namespace SLAM
