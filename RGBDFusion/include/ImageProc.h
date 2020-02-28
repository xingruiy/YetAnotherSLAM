#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

void RenderScene(const cv::cuda::GpuMat vmap,
                 const cv::cuda::GpuMat nmap,
                 cv::cuda::GpuMat &image);

void ComputeImageGradientCentralDifference(const cv::cuda::GpuMat image,
                                           cv::cuda::GpuMat &gx,
                                           cv::cuda::GpuMat &gy);

void TransformReferencePoint(const cv::cuda::GpuMat depth,
                             cv::cuda::GpuMat &vmap,
                             const Eigen::Matrix3d &K,
                             const Sophus::SE3d &T);

void DepthToInvDepth(const cv::cuda::GpuMat depth,
                     cv::cuda::GpuMat &invDepth);

void PyrDownDepth(const cv::cuda::GpuMat src,
                  cv::cuda::GpuMat &dst);