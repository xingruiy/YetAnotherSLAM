#pragma once

#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class RGBDTracking
{

public:
    RGBDTracking(const int &w,
                 const int &h,
                 const Eigen::Matrix3d &K,
                 const int &nNumPyr,
                 const std::vector<int> &vIterations,
                 const bool &bUseRGB,
                 const bool &bUseDepth);

    bool IsTrackingGood() const;
    void SwitchFrame();
    void SetReferenceImage(const cv::Mat &imGray);
    void SetReferenceDepth(const cv::Mat &imDepth);
    void SetTrackingImage(const cv::Mat &imGray);
    void SetTrackingDepth(const cv::Mat &imDepth);
    void SetReferenceInvD(cv::cuda::GpuMat imInvD);
    Eigen::Matrix<double, 6, 6> GetCovariance();

    Sophus::SE3d GetTransform(Sophus::SE3d estimate, const bool &bSwitchFrame = true);
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

    // Pyramid level and iterations
    const int mnNumPyr;
    std::vector<int> mvIterations;

    // Camera parameters
    std::vector<int> mvImageWidth;
    std::vector<int> mvImageHeight;
    std::vector<Eigen::Matrix3d> mK;

    // Tracking images
    std::vector<cv::cuda::GpuMat> mvCurrentDepth;
    std::vector<cv::cuda::GpuMat> mvReferenceDepth;
    std::vector<cv::cuda::GpuMat> mvCurrentIntensity;
    std::vector<cv::cuda::GpuMat> mvReferenceIntensity;
    std::vector<cv::cuda::GpuMat> mvIntensityGradientX;
    std::vector<cv::cuda::GpuMat> mvIntensityGradientY;
    std::vector<cv::cuda::GpuMat> mvReferencePointTransformed;
    std::vector<cv::cuda::GpuMat> mvCurrentInvDepth;
    std::vector<cv::cuda::GpuMat> mvReferenceInvDepth;
    std::vector<cv::cuda::GpuMat> mvInvDepthGradientX;
    std::vector<cv::cuda::GpuMat> mvInvDepthGradientY;

    float residualSum;
    float iResidualSum;
    float dResidualSum;
    float numResidual;

    Eigen::Matrix<float, 6, 6> hessian;
    Eigen::Matrix<float, 6, 1> residual;
};