#pragma once

#include "CudaUtils.h"
#include <cuda_runtime_api.h>

#define WarpSize 32

template <typename T, int size>
__device__ inline void WarpReduceSum(T *val)
{
#pragma unroll
    for (int offset = WarpSize / 2; offset > 0; offset /= 2)
    {
#pragma unroll
        for (int i = 0; i < size; ++i)
        {
            val[i] += __shfl_down_sync(0xffffffff, val[i], offset);
        }
    }
}

template <typename T, int size>
__device__ inline void BlockReduceSum(T *val)
{
    static __shared__ T shared[32 * size];
    int lane = threadIdx.x % WarpSize;
    int wid = threadIdx.x / WarpSize;

    WarpReduceSum<T, size>(val);

    if (lane == 0)
        memcpy(&shared[wid * size], val, sizeof(T) * size);

    __syncthreads();

    if (threadIdx.x < blockDim.x / WarpSize)
        memcpy(val, &shared[lane * size], sizeof(T) * size);
    else
        memset(val, 0, sizeof(T) * size);

    if (wid == 0)
        WarpReduceSum<T, size>(val);
}

template <int rows, int cols>
void inline RankUpdateHessian(float *hostData, float *hessian, float *residual)
{
    int shift = 0;
    for (int i = 0; i < rows; ++i)
        for (int j = i; j < cols; ++j)
        {
            float value = hostData[shift++];
            if (j == rows)
                residual[i] = value;
            else
                hessian[j * rows + i] = hessian[i * rows + j] = value;
        }
}

template <typename T>
__device__ __forceinline__ T interpolateBiLinear(
    const cv::cuda::PtrStep<T> &map,
    const float &x, const float &y)
{
    int u = static_cast<int>(std::floor(x));
    int v = static_cast<int>(std::floor(y));
    float cox = x - u;
    float coy = y - v;
    return (map.ptr(v)[u] * (1 - cox) + map.ptr(v)[u + 1] * cox) * (1 - coy) +
           (map.ptr(v + 1)[u] * (1 - cox) + map.ptr(v + 1)[u + 1] * cox) * coy;
}

struct se3StepRGBResidualFunctor
{
    int w, h, n;
    float fx, fy, cx, cy;
    cv::cuda::PtrStep<Eigen::Vector4f> refPtWarped;
    cv::cuda::PtrStep<float> refInt;
    cv::cuda::PtrStep<float> currInt;
    cv::cuda::PtrStep<float> currGx;
    cv::cuda::PtrStep<float> currGy;

    mutable cv::cuda::PtrStep<float> out;
    mutable cv::cuda::PtrStep<Eigen::Vector4f> refResidual;

    __device__ __forceinline__ bool findCorresp(
        const int &x, const int &y,
        float &residual,
        float &gx,
        float &gy) const
    {
        Eigen::Vector4f ptWarped = refPtWarped.ptr(y)[x];

        if (ptWarped(3) > 0)
        {
            float u = fx * ptWarped(0) / ptWarped(2) + cx;
            float v = fy * ptWarped(1) / ptWarped(2) + cy;

            if (u > 0 && v > 0 && u < w - 1 && v < h - 1)
            {
                auto refVal = refInt.ptr(y)[x];
                auto currVal = interpolateBiLinear(currInt, u, v);

                residual = currVal - refVal;
                gx = interpolateBiLinear(currGx, u, v);
                gy = interpolateBiLinear(currGy, u, v);

                return (gx * gx + gy * gy) > 0 && isfinite(residual);
            }
        }

        return false;
    }

    __device__ __forceinline__ void computeResidual(const int &k, float *res) const
    {
        const int y = k / w;
        const int x = k - y * w;

        float residual = 0.f;
        float gx, gy;

        bool correspFound = findCorresp(x, y, residual, gx, gy);

        res[0] = correspFound ? residual : 0.f;
        res[1] = (float)correspFound;

        refResidual.ptr(y)[x] = Eigen::Vector4f(residual, gx, gy, (float)correspFound - 0.5f);
    }

    __device__ __forceinline__ void operator()() const
    {
        float sum[2] = {0, 0};
        float res[2];
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            computeResidual(k, res);

            sum[0] += res[0];
            sum[1] += res[1];
        }

        BlockReduceSum<float, 2>(sum);

        if (threadIdx.x == 0)
        {
            out.ptr(blockIdx.x)[0] = sum[0];
            out.ptr(blockIdx.x)[1] = sum[1];
        }
    }
};

struct se3StepRGBFunctor
{
    int w, h, n;
    float fx, fy;
    float huberTh;
    cv::cuda::PtrStep<Eigen::Vector4f> refPtWarped;
    cv::cuda::PtrStep<Eigen::Vector4f> refResidual;

    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ void computeJacobian(const int &k, float *sum) const
    {
        const int y = k / w;
        const int x = k - y * w;
        const Eigen::Vector4f &res = refResidual.ptr(y)[x];
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        float wt = 1.0f;

        if (res(3) > 0)
        {
            Eigen::Vector3f pt = refPtWarped.ptr(y)[x].head<3>();
            float zInv = 1.0f / pt(2);

            if (abs(res(0)) > huberTh)
            {
                wt = huberTh / abs(res(0));
            }

            row[0] = res(1) * fx * zInv;
            row[1] = res(2) * fy * zInv;
            row[2] = -(row[0] * pt(0) + row[1] * pt(1)) * zInv;
            row[3] = row[2] * pt(1) - res(2) * fy;
            row[4] = -row[2] * pt(0) + res(1) * fx;
            row[5] = -row[0] * pt(1) + row[1] * pt(0);
            row[6] = -res(0);
        }

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
#pragma unroll
            for (int j = i; j < 7; ++j)
                sum[count++] = wt * row[i] * row[j];
    }

    __device__ __forceinline__ void operator()() const
    {
        float sum[29];
        memset(&sum[0], 0, sizeof(float) * 29);

        float temp[29];
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            computeJacobian(k, temp);
#pragma unroll
            for (int i = 0; i < 29; ++i)
                sum[i] += temp[i];
        }

        BlockReduceSum<float, 29>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }
};

struct se3StepDResidualFunctor
{
    int w, h, n;
    float fx, fy, cx, cy;
    cv::cuda::PtrStep<Eigen::Vector4f> refPtWarped;
    cv::cuda::PtrStep<float> refInvDepth;
    cv::cuda::PtrStep<float> currInvDepth;
    cv::cuda::PtrStep<float> currIDepthGx;
    cv::cuda::PtrStep<float> currIDepthGy;

    mutable cv::cuda::PtrStep<float> out;
    mutable cv::cuda::PtrStep<Eigen::Vector4f> refResidual;

    __device__ __forceinline__ bool findCorresp(
        const int &x, const int &y,
        float &residual,
        float &gx,
        float &gy) const
    {
        Eigen::Vector4f ptWarped = refPtWarped.ptr(y)[x];

        if (ptWarped(3) > 0)
        {
            float u = fx * ptWarped(0) / ptWarped(2) + cx;
            float v = fy * ptWarped(1) / ptWarped(2) + cy;

            if (u > 0 && v > 0 && u < w - 1 && v < h - 1)
            {
                residual = interpolateBiLinear(currInvDepth, u, v) - 1.0f / ptWarped(2);
                gx = interpolateBiLinear(currIDepthGx, u, v);
                gy = interpolateBiLinear(currIDepthGy, u, v);

                return (gx * gx + gy * gy) > 0 && isfinite(residual);
            }
        }

        return false;
    }

    __device__ __forceinline__ void computeResidual(const int &k, float *res) const
    {
        const int y = k / w;
        const int x = k - y * w;

        float residual = 0.f;
        float gx, gy;

        bool correspFound = findCorresp(x, y, residual, gx, gy);

        res[0] = correspFound ? residual : 0.f;
        res[1] = (float)correspFound;

        refResidual.ptr(y)[x] = Eigen::Vector4f(residual, gx, gy, (float)correspFound - 0.5f);
    }

    __device__ __forceinline__ void operator()() const
    {
        float sum[2] = {0, 0};
        float res[2];
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            computeResidual(k, res);

            sum[0] += res[0];
            sum[1] += res[1];
        }

        BlockReduceSum<float, 2>(sum);

        if (threadIdx.x == 0)
        {
            out.ptr(blockIdx.x)[0] = sum[0];
            out.ptr(blockIdx.x)[1] = sum[1];
        }
    }
};

struct se3StepDFunctor
{
    int w, h, n;
    float fx, fy;
    float huberTh;
    cv::cuda::PtrStep<Eigen::Vector4f> refPtWarped;
    cv::cuda::PtrStep<Eigen::Vector4f> refResidual;

    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ void computeJacobian(const int &k, float *sum) const
    {
        const int y = k / w;
        const int x = k - y * w;
        const Eigen::Vector4f &res = refResidual.ptr(y)[x];
        float row[7] = {0, 0, 0, 0, 0, 0, 0};
        float wt = 1.0f;

        if (res(3) > 0)
        {
            Eigen::Vector3f pt = refPtWarped.ptr(y)[x].head<3>();
            float zInv = 1.0f / pt(2);
            float zInvSqr = zInv * zInv;

            if (abs(res(0)) < huberTh)
            {
                wt = 1 - (res(0) / huberTh) * (res(0) / huberTh);
                wt *= wt;
            }
            else
                wt = 0.0f;

            row[0] = res(1) * fx * zInv;
            row[1] = res(2) * fy * zInv;
            row[2] = -(row[0] * pt(0) + row[1] * pt(1)) * zInv;
            row[3] = row[2] * pt(1) - res(2) * fy;
            row[4] = -row[2] * pt(0) + res(1) * fx;
            row[5] = -row[0] * pt(1) + row[1] * pt(0);
            row[6] = -res(0);
            row[2] += zInvSqr;
            row[3] += zInvSqr * pt(1);
            row[4] -= zInvSqr * pt(0);

            wt *= 1.0 / sqrt(res(1) * res(1) + res(2) * res(2));
            wt *= zInv;
        }

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
#pragma unroll
            for (int j = i; j < 7; ++j)
                sum[count++] = wt * row[i] * row[j];
    }

    __device__ __forceinline__ void operator()() const
    {
        float sum[29];
        memset(&sum[0], 0, sizeof(float) * 29);

        float temp[29];
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            computeJacobian(k, temp);
#pragma unroll
            for (int i = 0; i < 29; ++i)
                sum[i] += temp[i];
        }

        BlockReduceSum<float, 29>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }
};

struct se3StepRGBDResidualFunctor
{

    int w, h, n;
    float fx, fy, cx, cy;
    cv::cuda::PtrStep<Eigen::Vector4f> refPtWarped;
    cv::cuda::PtrStep<float> currInvDepth;
    cv::cuda::PtrStep<float> refInt;
    cv::cuda::PtrStep<float> currInt;
    cv::cuda::PtrStep<float> currGx;
    cv::cuda::PtrStep<float> currGy;
    cv::cuda::PtrStep<float> currInvDepthGx;
    cv::cuda::PtrStep<float> currInvDepthGy;

    mutable cv::cuda::PtrStep<float> out;
    mutable cv::cuda::PtrStep<Eigen::Matrix<float, 7, 1>> refResidual;

    __device__ __forceinline__ bool findCorresp(
        const int &x, const int &y,
        float &residual,
        float &residualD,
        float &gxZ,
        float &gyZ,
        float &gx,
        float &gy) const
    {
        Eigen::Vector4f ptWarped = refPtWarped.ptr(y)[x];

        if (ptWarped(3) > 0)
        {
            float u = fx * ptWarped(0) / ptWarped(2) + cx;
            float v = fy * ptWarped(1) / ptWarped(2) + cy;

            if (u > 0 && v > 0 && u < w - 1 && v < h - 1)
            {
                auto refVal = refInt.ptr(y)[x];
                auto currVal = interpolateBiLinear(currInt, u, v);

                residual = currVal - refVal;
                residualD = interpolateBiLinear(currInvDepth, u, v) - 1.0f / ptWarped(2);
                gxZ = interpolateBiLinear(currInvDepthGx, u, v);
                gyZ = interpolateBiLinear(currInvDepthGy, u, v);
                gx = interpolateBiLinear(currGx, u, v);
                gy = interpolateBiLinear(currGy, u, v);

                return (gx * gx + gy * gy) > 0 &&
                       (gxZ * gxZ + gyZ * gyZ) > 0 &&
                       isfinite(residual) &&
                       isfinite(residualD);
            }
        }

        return false;
    }

    __device__ __forceinline__ void computeResidual(const int &k, float *res) const
    {
        const int y = k / w;
        const int x = k - y * w;

        float residual = 0.f;
        float residualD = 0.f;
        float gx, gy, gxZ, gyZ;

        bool correspFound = findCorresp(x, y, residual, residualD, gxZ, gyZ, gx, gy);

        res[0] = correspFound ? residual : 0.f;
        res[1] = correspFound ? residualD : 0.f;
        res[2] = (float)correspFound;

        refResidual.ptr(y)[x] = Eigen::Matrix<float, 7, 1>(residual, residualD, gx, gy, gxZ, gyZ, (float)correspFound - 0.5f);
    }

    __device__ __forceinline__ void operator()() const
    {
        float sum[3] = {0, 0, 0};
        float res[3];
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            computeResidual(k, res);

#pragma unroll
            for (int i = 0; i < 3; ++i)
                sum[i] += res[i];
        }

        BlockReduceSum<float, 3>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 3; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }
};

struct se3StepRGBDFunctor
{
    int w, h, n;
    float fx, fy;
    Eigen::Matrix2f precision;
    float stddevI, stddevD;
    cv::cuda::PtrStep<Eigen::Vector4f> refPtWarped;
    cv::cuda::PtrStep<Eigen::Matrix<float, 7, 1>> refResidual;

    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ void computeJacobian(const int &k, float *sum) const
    {
        const int y = k / w;
        const int x = k - y * w;
        const Eigen::Matrix<float, 7, 1> &res = refResidual.ptr(y)[x];
        Eigen::Matrix<float, 2, 6> J = Eigen::Matrix<float, 2, 6>::Zero();
        Eigen::Vector2f r = Eigen::Vector2f::Zero();
        float wt = 1.0f;
        float wtc = 1.0f;
        float wtd = 1.0f;

        if (res(6) > 0)
        {
            Eigen::Vector3f pt = refPtWarped.ptr(y)[x].head<3>();
            float zInv = 1.0f / pt(2);
            float zInvSqr = zInv * zInv;

            if (abs(res(0)) > stddevI)
            {
                wtc *= stddevI / abs(res(0));
                wtc *= 1.0f / (1.0f + res(2) * res(2) + res(3) * res(3));
            }

            if (abs(res(1)) > stddevD)
            {
                wtd *= stddevD / abs(res(1));
                wtd *= 1.0f / (1.0f + res(4) * res(4) + res(5) * res(5));
            }

            r(0, 0) = -res(0);
            r(1, 0) = -res(1);

            wt *= 6 / (5 + r.transpose() * precision * r);
            wt *= zInv;

            J(0, 0) = res(2) * fx * zInv;
            J(0, 1) = res(3) * fy * zInv;
            J(0, 2) = -(J(0, 0) * pt(0) + J(0, 1) * pt(1)) * zInv;
            J(0, 3) = J(0, 2) * pt(1) - res(3) * fy;
            J(0, 4) = -J(0, 2) * pt(0) + res(2) * fx;
            J(0, 5) = -J(0, 0) * pt(1) + J(0, 1) * pt(0);
            J.row(0) = wtc * J.row(0);

            J(1, 0) = res(4) * fx * zInv;
            J(1, 1) = res(5) * fy * zInv;
            J(1, 2) = -(J(1, 0) * pt(0) + J(1, 1) * pt(1)) * zInv;
            J(1, 3) = J(1, 2) * pt(1) - res(5) * fy;
            J(1, 4) = -J(1, 2) * pt(0) + res(4) * fx;
            J(1, 5) = -J(1, 0) * pt(1) + J(1, 1) * pt(0);
            J(1, 2) += zInvSqr;
            J(1, 3) += zInvSqr * pt(1);
            J(1, 4) -= zInvSqr * pt(0);
            J.row(1) = wtd * J.row(1);
        }

        Eigen::Matrix<float, 6, 6> hessian;
        hessian = wt * J.transpose() * precision * J;
        Eigen::Matrix<float, 6, 1> residual = wt * J.transpose() * precision * r;

        int counter = 0;

#pragma unroll
        for (int row = 0; row < 6; ++row)
#pragma unroll
            for (int col = row; col < 7; ++col)
                if (col == 6)
                    sum[counter++] = residual(row);
                else
                    sum[counter++] = hessian(row, col);
        sum[counter] = wt * r.transpose() * precision * r;
    }

    __device__ __forceinline__ void operator()() const
    {
        float sum[29];
        memset(&sum[0], 0, sizeof(float) * 29);

        float temp[29];
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            computeJacobian(k, temp);
#pragma unroll
            for (int i = 0; i < 29; ++i)
                sum[i] += temp[i];
        }

        BlockReduceSum<float, 29>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }
};

struct VarianceEstimator
{
    int w, h, n;
    float meanEstimated;
    cv::cuda::PtrStep<Eigen::Vector4f> residual;

    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ void operator()() const
    {
        float sum[1] = {0};
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            int y = k / w;
            int x = k - y * w;

            const Eigen::Vector4f &res = residual.ptr(y)[x];

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
    Eigen::Vector2f meanEstimated;
    cv::cuda::PtrStep<Eigen::Matrix<float, 7, 1>> residual;

    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ void operator()() const
    {
        float sum[3] = {0, 0, 0};
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
        {
            int y = k / w;
            int x = k - y * w;

            const Eigen::Matrix<float, 7, 1> &res = residual.ptr(y)[x];

            if (res(6) > 0)
            {
                Eigen::Vector2f var = Eigen::Vector2f(res(0), res(1)) - meanEstimated;
                Eigen::Matrix2f varCovMat = var * var.transpose();

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
    cv::cuda::PtrStep<Eigen::Vector4f> ptWarped;
    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ void getOpticalFlow(const int &k, float *val) const
    {
        int y = k / w;
        int x = k - y * w;

        val[0] = 0;
        val[1] = 0;

        Eigen::Vector4f pt = ptWarped.ptr(y)[x];
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