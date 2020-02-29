#pragma once

#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#define NUM_PYR 5

class RGBDTracking
{

public:
    RGBDTracking(const int &w, const int &h, const Eigen::Matrix3d &K,
                 const bool &bUseRGB, const bool &bUseDepth);

    bool IsTrackingGood() const;

    void SetReferenceImage(const cv::Mat &imGray);
    void SetReferenceDepth(const cv::Mat &imDepth);

    void SetTrackingImage(const cv::Mat &imGray);
    void SetTrackingDepth(const cv::Mat &imDepth);

    void SetReferenceInvD(cv::cuda::GpuMat imInvD);

    Sophus::SE3d GetTransform(const Sophus::SE3d &init, const bool bSwapBuffer);
    void SwapFrameBuffer();

    cv::cuda::GpuMat GetReferenceDepth(const int lvl = 0) const;

private:
    enum class TrackingModal
    {
        RGB_ONLY,
        DEPTH_ONLY,
        RGB_AND_DEPTH,
        IDLE
    };

    TrackingModal mModal;

    void ComputeSingleStepRGB(const int lvl, const Sophus::SE3d &T, float *hessian, float *residual);
    void ComputeSingleStepRGBDLinear(const int lvl, const Sophus::SE3d &T, float *hessian, float *residual);
    void ComputeSingleStepRGBD(const int lvl, const Sophus::SE3d &T, float *hessian, float *residual);
    void ComputeSingleStepDepth(const int lvl, const Sophus::SE3d &T, float *hessian, float *residual);
    void TransformReferencePoint(const int lvl, const Sophus::SE3d &estimate);

    // GPU buffer for temporary data
    cv::cuda::GpuMat mGpuBufferFloat96x29;
    cv::cuda::GpuMat mGpuBufferFloat96x3;
    cv::cuda::GpuMat mGpuBufferFloat96x2;
    cv::cuda::GpuMat mGpuBufferFloat96x1;
    cv::cuda::GpuMat mGpuBufferFloat1x29;
    cv::cuda::GpuMat mGpuBufferFloat1x3;
    cv::cuda::GpuMat mGpuBufferFloat1x2;
    cv::cuda::GpuMat mGpuBufferFloat1x1;
    cv::cuda::GpuMat mGpuBufferVector4HxW;
    cv::cuda::GpuMat mGpuBufferVector7HxW;

    // Dedicated buffer for temporary images
    cv::cuda::GpuMat mGpuBufferRawDepth;

    // Indicator result
    bool mbTrackingGood;

    // Camera parameters
    int mvWidth[NUM_PYR];
    int mvHeight[NUM_PYR];
    Eigen::Matrix3d mK[NUM_PYR];

    // Tracking images
    cv::cuda::GpuMat mvCurrentDepth[NUM_PYR];
    cv::cuda::GpuMat mvReferenceDepth[NUM_PYR];
    cv::cuda::GpuMat mvCurrentIntensity[NUM_PYR];
    cv::cuda::GpuMat mvReferenceIntensity[NUM_PYR];
    cv::cuda::GpuMat mvIntensityGradientX[NUM_PYR];
    cv::cuda::GpuMat mvIntensityGradientY[NUM_PYR];
    cv::cuda::GpuMat mvCurrentInvDepth[NUM_PYR];
    cv::cuda::GpuMat mvReferenceInvDepth[NUM_PYR];
    cv::cuda::GpuMat mvInvDepthGradientX[NUM_PYR];
    cv::cuda::GpuMat mvInvDepthGradientY[NUM_PYR];
    cv::cuda::GpuMat mvReferencePointTransformed[NUM_PYR];

    float residualSum;
    float iResidualSum;
    float dResidualSum;
    float numResidual;
};
