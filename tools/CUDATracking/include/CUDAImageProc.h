#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

void RenderScene(const cv::cuda::GpuMat vmap,
                 const cv::cuda::GpuMat nmap,
                 cv::cuda::GpuMat &image);

void BackprojectDepth(const cv::cuda::GpuMat depth,
                      cv::cuda::GpuMat &vmap,
                      const Eigen::Matrix3d &K);

void ComputeNMAP(const cv::cuda::GpuMat vmap,
                 cv::cuda::GpuMat &nmap);

void ComputeImageGradientCentralDifference(const cv::cuda::GpuMat image,
                                           cv::cuda::GpuMat &gx,
                                           cv::cuda::GpuMat &gy);

void TransformReferencePoint(const cv::cuda::GpuMat depth,
                             cv::cuda::GpuMat &vmap,
                             const Eigen::Matrix3d &K,
                             const Sophus::SE3d &T);

void ConvertDepthToInvDepth(const cv::cuda::GpuMat depth,
                            cv::cuda::GpuMat &invDepth);

void ConvertVMAPToInvDepth(const cv::cuda::GpuMat vmap,
                           cv::cuda::GpuMat &invDepth);

void PyrDownInvDepth(const cv::cuda::GpuMat src,
                     cv::cuda::GpuMat &dst);
