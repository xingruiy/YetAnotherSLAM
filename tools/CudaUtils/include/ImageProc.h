#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class ImageProc
{
public:
    static void BackProjectDepth(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &vmap);
};