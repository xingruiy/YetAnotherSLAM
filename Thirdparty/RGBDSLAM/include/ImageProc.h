#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

void renderScene(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, cv::cuda::GpuMat &image);
void computeVMap(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &vmap, const Eigen::Matrix3d &K);
void computeNormal(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &nmap);
void computeImageGradientCentralDiff(cv::cuda::GpuMat image, cv::cuda::GpuMat &gx, cv::cuda::GpuMat &gy);
void TransformReferencePoint(cv::cuda::GpuMat depth, cv::cuda::GpuMat &vmap, const Eigen::Matrix3d &K, const Sophus::SE3d &T);
void convertDepthToInvDepth(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &invDepth);
void convertVMapToInvDepth(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &invDepth);
void pyrdownInvDepth(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);