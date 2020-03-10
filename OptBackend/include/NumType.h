#ifndef NUM_TYPE_H
#define NUM_TYPE_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#define NUM_LOCAL_KF 7
#define MAX_NUM_RES 8

struct FramePoint
{
    float intensity;
    float invd;
    char host;
    Eigen::Matrix<char, MAX_NUM_RES, 1> target;
    Eigen::Matrix<float, MAX_NUM_RES, 1> residual;
};

struct FrameShell
{
    cv::cuda::GpuMat depth;
    cv::cuda::GpuMat image;
    cv::cuda::GpuMat dx, dy;

    FramePoint *points;
    Sophus::SE3f pose;
    Sophus::SE3f poseInv;
    float *framePosePrecalc;
};

#endif