#include "cudaDeviceFuncs.h"

struct LinearizeResidualFunc
{
    int w, h, N;
    cv::cuda::PtrStep<float> refImg, currImg;
    cv::cuda::PtrStep<float> gx, gy;
    cv::cuda::PtrStep<float> refGrad2;
    float fx, fy, cx, cy;
    float grad2Th;

    mutable cv::cuda::PtrStep<float> out;
    mutable cv::cuda::PtrStep<Eigen::Vector<float, 6>> buffer;

    __device__ __inline__ void operator()() const
    {
    }
};

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
                       cv::cuda::GpuMat &resOut)
{
}

void CalcHessianColour(cv::cuda::GpuMat &buffer,
                       const float fx,
                       const float fy,
                       float *hostH,
                       float *hostb)
{
}