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
    cv::cuda::PtrStep<Vec3f> dstPt)
{

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x >= adjacencyMat.cols)
        return;

    int y = x;
    float score = 0;

    if (x == y)
    {
        score = exp(-dist.ptr(0)[x]);
    }
    else
    {
    }

    adjacencyMat.ptr(y)[x] = score;
}

Mat createAdjacencyMat(
    size_t numPairs,
    Mat descriptorDist,
    Mat srcPointPos,
    Mat dstPointPos)
{
    GMat adjacencyMat(numPairs, numPairs, CV_32FC1);
    GMat gpuDescDist(descriptorDist);
    GMat gpuSrcPointPos(srcPointPos);
    GMat gpuDstPointPos(dstPointPos);

    dim3 block(8);
    dim3 grid(div_up(numPairs, block.x));
    createAdjacencyMatKernel<<<grid, block>>>(
        adjacencyMat,
        gpuDescDist,
        gpuSrcPointPos,
        gpuDstPointPos);

    Mat cpuAdjacencyMat(adjacencyMat);
    return cpuAdjacencyMat;
}