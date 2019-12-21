#pragma once
#define WarpSize 32

#include <cuda_runtime_api.h>

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