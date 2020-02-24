#pragma once

#include "reduceSum.h"
#include "cudaUtils.h"
#include <cuda_runtime_api.h>

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
