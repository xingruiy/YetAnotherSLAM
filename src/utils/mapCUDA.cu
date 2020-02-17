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
        // adjacencyMat.ptr(x)[y] = 1000;
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

void createAdjacencyMat(
    size_t numPairs,
    Mat descriptorDist,
    Mat srcPointPos,
    Mat dstPointPos,
    Mat validPairPt,
    Mat &adjacentMat)
{
    GMat gpuAdjacencyMat(numPairs, numPairs, CV_32FC1);
    GMat gpuDescDist(descriptorDist);
    GMat gpuSrcPointPos(srcPointPos);
    GMat gpuDstPointPos(dstPointPos);
    GMat gpuValidPtPair(validPairPt);

    dim3 block(8, 8);
    dim3 grid(div_up(numPairs, block.x), div_up(numPairs, block.y));

    createAdjacencyMatKernel<<<grid, block>>>(
        gpuAdjacencyMat,
        gpuDescDist,
        gpuSrcPointPos,
        gpuDstPointPos,
        gpuValidPtPair);

    adjacentMat = Mat(gpuAdjacencyMat);
}

__global__ void createAdjacencyMatKernel(
    cv::cuda::PtrStepSz<float> adjacencyMat,
    cv::cuda::PtrStepSz<float> dist,
    cv::cuda::PtrStep<Vec3f> srcPt,
    cv::cuda::PtrStep<Vec3f> dstPt,
    cv::cuda::PtrStep<Vec3f> srcNormal,
    cv::cuda::PtrStep<Vec3f> dstNormal,
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
        // adjacencyMat.ptr(x)[y] = 1000;
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

        const auto srcVec = srcPt1 - srcPt0;
        const auto dstVec = dstPt1 - dstPt0;
        float srcPtDist = srcVec.norm();
        float dstPtDist = dstVec.norm();

        if (srcPtDist <= FLT_EPSILON || dstPtDist <= FLT_EPSILON)
            score = 0;
        else
        {
            const auto &srcNm0 = srcNormal[x];
            const auto &srcNm1 = srcNormal[y];
            const auto &dstNm0 = dstNormal[x];
            const auto &dstNm1 = dstNormal[y];

            const auto srcAlpha = srcNm0.dot(srcNm1);
            const auto dstAlpha = dstNm0.dot(dstNm1);

            score = exp(-fabs(srcPtDist - dstPtDist) -
                        fabs(srcAlpha - dstAlpha));
        }
    }

    adjacencyMat.ptr(x)[y] = score;
    adjacencyMat.ptr(y)[x] = score;
}

void createAdjacencyMatWithNormal(
    const size_t numPairs,
    const Mat descriptorDist,
    const Mat srcPointPos,
    const Mat dstPointPos,
    const Mat srcPtNormal,
    const Mat dstPtNormal,
    const Mat validPairPt,
    Mat &adjacentMat)
{
    GMat gpuAdjacencyMat(numPairs, numPairs, CV_32FC1);
    GMat gpuDescDist(descriptorDist);
    GMat gpuSrcPointPos(srcPointPos);
    GMat gpuDstPointPos(dstPointPos);
    GMat gpuSrcPtNormal(srcPtNormal);
    GMat gpuDstPtNormal(dstPtNormal);
    GMat gpuValidPtPair(validPairPt);

    dim3 block(8, 8);
    dim3 grid(div_up(numPairs, block.x), div_up(numPairs, block.y));

    createAdjacencyMatKernel<<<grid, block>>>(
        gpuAdjacencyMat,
        gpuDescDist,
        gpuSrcPointPos,
        gpuDstPointPos,
        gpuSrcPtNormal,
        gpuDstPtNormal,
        gpuValidPtPair);

    adjacentMat = Mat(gpuAdjacencyMat);
}