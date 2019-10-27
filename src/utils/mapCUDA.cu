#include "utils/mapCUDA.h"
#include "utils/cudaUtils.h"

__device__ __forceinline__ float clamp(float a)
{
    a = a > -1.f ? a : -1.f;
    a = a < 1.f ? a : 1.f;
    return a;
}

__global__ void createAdjacencyMatKernel(
    cv::cuda::PtrStepSz<float> adjacencyMat,
    cv::cuda::PtrStepSz<float> dist,
    cv::cuda::PtrStep<Vec3f> srcPt,
    cv::cuda::PtrStep<Vec3f> dstPt,
    cv::cuda::PtrStep<bool> valid)
{

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= adjacencyMat.cols || y >= adjacencyMat.rows)
        return;

    float score = 0;
    const bool &isValid = valid.ptr(0)[x];
    if (!isValid)
    {
        return;
    }

    if (x == y)
    {
        score = exp(-dist.ptr(0)[x]);
    }
    else
    {
        const auto &srcPt0 = srcPt[x];
        const auto &srcPt1 = srcPt[y];
        const auto &dstPt0 = dstPt[x];
        const auto &dstPt1 = dstPt[y];

        float srcPtDist = (srcPt0 - srcPt1).norm();
        float dstPtDist = (dstPt0 - dstPt1).norm();

        if (srcPtDist <= FLT_EPSILON || dstPtDist <= FLT_EPSILON)
            score = 0;
        else
            score = exp(-fabs(srcPtDist - dstPtDist));
    }

    adjacencyMat.ptr(y)[x] = score;
    adjacencyMat.ptr(x)[y] = score;
}

Mat createAdjacencyMat(
    size_t numPairs,
    Mat descriptorDist,
    Mat srcPointPos,
    Mat dstPointPos,
    Mat validPairPt)
{
    GMat adjacencyMat(numPairs, numPairs, CV_32FC1);
    GMat gpuDescDist(descriptorDist);
    GMat gpuSrcPointPos(srcPointPos);
    GMat gpuDstPointPos(dstPointPos);
    GMat gpuValidPtPair(validPairPt);

    dim3 block(8, 8);
    dim3 grid(div_up(numPairs, block.x), div_up(numPairs, block.y));
    createAdjacencyMatKernel<<<grid, block>>>(
        adjacencyMat,
        gpuDescDist,
        gpuSrcPointPos,
        gpuDstPointPos,
        gpuValidPtPair);

    Mat cpuAdjacencyMat(adjacencyMat);
    return cpuAdjacencyMat;
}