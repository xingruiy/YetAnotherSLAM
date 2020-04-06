#ifndef CUDA_DEVICE_FUNCS_H
#define CUDA_DEVICE_FUNCS_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

void LinearizeResidual(const cv::cuda::GpuMat refDepth,
                       const cv::cuda::GpuMat currImg,
                       const cv::cuda::GpuMat refImg,
                       const float fx,
                       const float fy,
                       const float cx,
                       const float cy,
                       const float imgGradTh,
                       cv::cuda::GpuMat &buffer,
                       cv::cuda::GpuMat &resSum,
                       cv::cuda::GpuMat &resOut);

void LinearizeResidual(const cv::cuda::GpuMat vmapRef,
                       const cv::cuda::GpuMat vmapCurr,
                       const cv::cuda::GpuMat nmapRef,
                       const cv::cuda::GpuMat nmapCurr,
                       const float fx,
                       const float fy,
                       const float cx,
                       const float cy,
                       const float distTh,
                       const float angleTh,
                       cv::cuda::GpuMat &buffer,
                       cv::cuda::GpuMat &resSum,
                       cv::cuda::GpuMat &resOut);

void CalcHessianColour(cv::cuda::GpuMat &buffer,
                       const float fx,
                       const float fy,
                       float *hostH,
                       float *hostb);

void CalcHessianDepth(cv::cuda::GpuMat &buffer,
                      const float fx,
                      const float fy,
                      float *hostH,
                      float *hostb);

#endif