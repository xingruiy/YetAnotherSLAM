#include "cudaDeviceFuncs.h"

#define Tid blockDim.x *blockIdx.x + threadIdx.x
#define GridStride gridDim.x *blockDim.x

__global__ void MakeImageVec3_kernel(float *src, Eigen::Vector3f *dst, int N)
{
    for (int idx = Tid; idx < N; idx += GridStride)
    {
        dst[idx][0] = src[idx];
    }
}

void MakeImageVec3(float *img_src, Eigen::Vector3f *img, int w, int h)
{
    dim3 block(1024);
    dim3 grid((w * h + block.x - 1) / block.x);

    MakeImageVec3_kernel<<<grid, block>>>(img_src, img, w * h);
}

__global__ void MakeDepthVec3_kernel(float *src, Eigen::Vector3f *dst, int N)
{
    for (int idx = Tid; idx < N; idx += GridStride)
    {
        float idepth = 1.0f / src[idx];
        if (isfinite(idepth) && !isnan(idepth))
            dst[idx][0] = idepth;
        else
            dst[idx][0] = 0;
    }
}

void MakeDepthVec3(float *depth_src, Eigen::Vector3f *depth, int w, int h)
{
    dim3 block(1024);
    dim3 grid((w * h + block.x - 1) / block.x);

    MakeImageVec3_kernel<<<grid, block>>>(depth_src, depth, w * h);
}

__global__ void PyraDownImage_kernel(Eigen::Vector3f *img_src, Eigen::Vector3f *img_dst, int w, int h, int N)
{
    int wlm1 = w * 2;
    for (int idx = Tid; idx < N; idx += GridStride)
    {
        int y = idx / w;
        int x = idx - y * w;
        int idxlm1 = y * 2 * wlm1 + x * 2;

        img_dst[idx][0] = (img_src[idxlm1][0] +
                           img_src[idxlm1 + 1][0] +
                           img_src[idxlm1 + wlm1][0] +
                           img_src[idxlm1 + wlm1 + 1][0]) *
                          0.25f;
    }
}

void PyraDownImage(Eigen::Vector3f *img_src, Eigen::Vector3f *img_dst, int w, int h)
{
    dim3 block(1024);
    dim3 grid((w * h + block.x - 1) / block.x);

    PyraDownImage_kernel<<<grid, block>>>(img_src, img_dst, w, h, w * h);
}

__global__ void MakeImageGradients_kernel(Eigen::Vector3f *img, float *grad2, int w, int h, int N)
{
    for (int idx = Tid; idx < N; idx += GridStride)
    {
        float dx = 0, dy = 0;
        if (!(idx == 1 || idx / w == h - 1))
        {
            float dx = 0.5f * (img[idx + 1][0] - img[idx - 1][0]);
            float dy = 0.5f * (img[idx + w][0] - img[idx - w][0]);
        }

        img[idx][1] = dx;
        img[idx][2] = dy;
        grad2[idx] = dx * dx + dy * dy;
    }
}

void MakeImageGradients(Eigen::Vector3f *img, float *grad2, int w, int h)
{
    dim3 block(1024);
    dim3 grid((w * h + block.x - 1) / block.x);

    MakeImageGradients_kernel<<<grid, block>>>(img, grad2, w, h, w * h);
}

void PointSelection(Eigen::Vector3f *img, float *grad2, int w, int h)
{
}