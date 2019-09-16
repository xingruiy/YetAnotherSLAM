#pragma once
#include "utils/numType.h"
#include "utils/reduceSum.h"
#include <cuda_runtime_api.h>

struct VarianceEstimator
{
    int w, h, n;
    float meanEstimated;
    cv::cuda::PtrStep<Vec4f> residual;

    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ void operator()() const
    {
        float sum[1] = {0};
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            int y = k / w;
            int x = k - y * w;

            const Vec4f &res = residual.ptr(y)[x];

            if (res(3) > 0)
            {
                sum[0] += ((res(0) - meanEstimated) * (res(0) - meanEstimated));
            }
        }

        BlockReduceSum<float, 1>(sum);

        if (threadIdx.x == 0)
            out.ptr(blockIdx.x)[0] = sum[0];
    }
};

struct VarCov2DEstimator
{
    int w, h, n;
    Vec2f meanEstimated;
    cv::cuda::PtrStep<Vec7f> residual;

    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ void operator()() const
    {
        float sum[3] = {0, 0, 0};
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            int y = k / w;
            int x = k - y * w;

            const Vec7f &res = residual.ptr(y)[x];

            if (res(6) > 0)
            {
                Vec2f var = Vec2f(res(0), res(1)) - meanEstimated;
                Mat22f varCovMat = var * var.transpose();

                sum[0] += varCovMat(0, 0);
                sum[1] += varCovMat(1, 1);
                sum[2] += varCovMat(0, 1);
            }
        }

        BlockReduceSum<float, 3>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 3; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }
};

struct computeOpticalFlowFunctor
{
    int w, h, n;
    float fx, fy, cx, cy;
    cv::cuda::PtrStep<Vec4f> ptWarped;
    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ void getOpticalFlow(const int &k, float *val) const
    {
        int y = k / w;
        int x = k - y * w;

        val[0] = 0;
        val[1] = 0;

        Vec4f pt = ptWarped.ptr(y)[x];
        if (pt(3) > 0)
        {
            float u = fx * pt(0) / pt(2) + cx;
            float v = fy * pt(1) / pt(2) + cy;
            if (u > 0 && v > 0 && u < w - 1 && v < h - 1)
            {
                val[0] = (x - u) * (x - u) / (w * w) + (y - v) * (y - v) / (h * h);
                val[1] = 1.f;
            }
        }
    }

    __device__ __forceinline__ void operator()() const
    {
        float sum[2] = {0, 0};
        float val[2];
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            getOpticalFlow(k, val);
            sum[0] += val[0];
            sum[1] += val[1];
        }

        BlockReduceSum<float, 2>(sum);

        if (threadIdx.x == 0)
        {
            out.ptr(blockIdx.x)[0] = sum[0];
            out.ptr(blockIdx.x)[1] = sum[1];
        }
    }
};