#include "ImageProc.h"
#include "utils/CudaUtils.h"

#define DEPTH_MAX 8.f
#define DEPTH_MIN 0.2f

__global__ void ComputeImageGradientCentralDifference_kernel(const cv::cuda::PtrStepSz<float> src,
                                                             cv::cuda::PtrStep<float> gx,
                                                             cv::cuda::PtrStep<float> gy)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x > src.cols - 1 || y > src.rows - 1)
        return;

    if (x < 1 || y < 1 || x > src.cols - 2 || y > src.rows - 2)
    {
        gx.ptr(y)[x] = gy.ptr(y)[x] = 0;
    }
    else
    {
        gx.ptr(y)[x] = (src.ptr(y)[x + 1] - src.ptr(y)[x - 1]) * 0.5f;
        gy.ptr(y)[x] = (src.ptr(y + 1)[x] - src.ptr(y - 1)[x]) * 0.5f;
    }
}

void ComputeImageGradientCentralDifference(const cv::cuda::GpuMat image,
                                           cv::cuda::GpuMat &gx,
                                           cv::cuda::GpuMat &gy)
{
    if (gx.empty())
        gx.create(image.size(), CV_32FC1);
    if (gy.empty())
        gy.create(image.size(), CV_32FC1);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(image.cols, block.x), cv::divUp(image.rows, block.y));

    ComputeImageGradientCentralDifference_kernel<<<grid, block>>>(image, gx, gy);
}

__global__ void TransformReferencePoint_kernel(const cv::cuda::PtrStepSz<float> depth,
                                               cv::cuda::PtrStep<Eigen::Vector4f> vTrans,
                                               Eigen::Matrix3f RKinv,
                                               Eigen::Vector3f t)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= depth.cols || y >= depth.rows)
        return;

    const float &zInv = depth.ptr(y)[x];
    if (zInv > 0.f)
    {
        float z = 1.0 / zInv;
        Eigen::Vector3f pt = RKinv * Eigen::Vector3f(x, y, 1.0f) * z + t;
        vTrans.ptr(y)[x] = Eigen::Vector4f(pt(0), pt(1), pt(2), 1.0f);
    }
    else
        vTrans.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1.0f);
}

void TransformReferencePoint(const cv::cuda::GpuMat depth,
                             cv::cuda::GpuMat &vmap,
                             const Eigen::Matrix3d &K,
                             const Sophus::SE3d &T)
{
    if (vmap.empty())
        vmap.create(depth.size(), CV_32FC4);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(depth.cols, block.x), cv::divUp(depth.rows, block.y));

    Eigen::Matrix3d RKinv = T.matrix().topLeftCorner(3, 3) * K.inverse();
    Eigen::Vector3d t = T.matrix().topRightCorner(3, 1);

    TransformReferencePoint_kernel<<<grid, block>>>(depth, vmap, RKinv.cast<float>(), t.cast<float>());
}

__device__ __forceinline__ Eigen::Vector<uchar, 4> RenderPoint(const Eigen::Vector3f &point,
                                                               const Eigen::Vector3f &normal,
                                                               const Eigen::Vector3f &image,
                                                               const Eigen::Vector3f &lightPos)
{
    Eigen::Vector3f colour(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
    if (!isnan(point(0)))
    {
        const float Ka = 0.3f;     // ambient coeff
        const float Kd = 0.5f;     // diffuse coeff
        const float Ks = 0.2f;     // specular coeff
        const float n = 20.f;      // specular power
        const float Ax = image(0); // ambient color
        const float Dx = image(1); // diffuse color
        const float Sx = image(2); // specular color
        const float Lx = 1.f;      // light color

        Eigen::Vector3f L = (lightPos - point).normalized();
        Eigen::Vector3f V = (Eigen::Vector3f(0.f, 0.f, 0.f) - point).normalized();
        Eigen::Vector3f R = (2 * normal * (normal.dot(L)) - L).normalized();

        float Ix = Ax * Ka * Dx + Lx * Kd * Dx * fmax(0.f, (normal.dot(L))) + Lx * Ks * Sx * pow(fmax(0.f, (R.dot(V))), n);
        colour = Eigen::Vector3f(Ix, Ix, Ix);
    }

    return Eigen::Vector<uchar, 4>(static_cast<uchar>(__saturatef(colour(0)) * 255.f),
                                   static_cast<uchar>(__saturatef(colour(1)) * 255.f),
                                   static_cast<uchar>(__saturatef(colour(2)) * 255.f),
                                   255);
}

__global__ void RenderScene_kernel(const cv::cuda::PtrStep<Eigen::Vector4f> vmap,
                                   const cv::cuda::PtrStep<Eigen::Vector4f> nmap,
                                   const Eigen::Vector3f lightPos,
                                   cv::cuda::PtrStepSz<Eigen::Vector<uchar, 4>> dst)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= dst.cols || y >= dst.rows)
        return;

    Eigen::Vector3f point = vmap.ptr(y)[x].head<3>();
    Eigen::Vector3f normal = nmap.ptr(y)[x].head<3>();
    Eigen::Vector3f pixel(1.f, 1.f, 1.f);

    dst.ptr(y)[x] = RenderPoint(point, normal, pixel, lightPos);
}

void RenderScene(const cv::cuda::GpuMat vmap,
                 const cv::cuda::GpuMat nmap,
                 cv::cuda::GpuMat &image)
{
    if (image.empty())
        image.create(vmap.size(), CV_8UC4);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

    RenderScene_kernel<<<grid, block>>>(vmap, nmap, Eigen::Vector3f(5, 5, 5), image);
}

__global__ void DepthToInvDepth_kernel(const cv::cuda::PtrStep<float> depth, cv::cuda::PtrStepSz<float> depth_inv)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x > depth_inv.cols - 1 || y > depth_inv.rows - 1)
        return;

    const float z = depth.ptr(y)[x];
    if (z > DEPTH_MIN && z < DEPTH_MAX)
        depth_inv.ptr(y)[x] = 1.0 / z;
    else
        depth_inv.ptr(y)[x] = 0;
}

void DepthToInvDepth(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &depth_inv)
{
    if (depth_inv.empty())
        depth_inv.create(depth.size(), depth.type());

    dim3 block(8, 8);
    dim3 grid(cv::divUp(depth.cols, block.x), cv::divUp(depth.rows, block.y));

    DepthToInvDepth_kernel<<<grid, block>>>(depth, depth_inv);
}

__global__ void PyrDownDepth_kernel(const cv::cuda::PtrStep<float> src,
                                    cv::cuda::PtrStepSz<float> dst)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= dst.cols - 1 || y >= dst.rows - 1)
        return;

    dst.ptr(y)[x] = src.ptr(2 * y)[2 * x];
}

void PyrDownDepth(const cv::cuda::GpuMat src,
                  cv::cuda::GpuMat &dst)
{
    if (dst.empty())
        dst.create(src.size(), CV_32FC1);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(src.cols, block.x), cv::divUp(src.rows, block.y));

    PyrDownDepth_kernel<<<grid, block>>>(src, dst);
}

__global__ void ComputeVertexMap_kernel(const cv::cuda::PtrStepSz<float> depth_inv,
                                        cv::cuda::PtrStep<Eigen::Vector4f> vmap,
                                        const float invfx, const float invfy,
                                        const float cx, const float cy, const float cut_off)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= depth_inv.cols || y >= depth_inv.rows)
        return;

    const float invz = depth_inv.ptr(y)[x];
    const float z = 1.0 / invz;
    if (invz > 0 && z < cut_off)
    {
        vmap.ptr(y)[x] = Eigen::Vector4f(z * (x - cx) * invfx, z * (y - cy) * invfy, z, 1.0);
    }
    else
    {
        vmap.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1.f);
    }
}

void ComputeVertexMap(const cv::cuda::GpuMat depth_inv, cv::cuda::GpuMat vmap, const float invfx, const float invfy, const float cx, const float cy, const float cut_off)
{
    if (vmap.empty())
        vmap.create(depth_inv.size(), CV_32FC4);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(depth_inv.cols, block.x), cv::divUp(depth_inv.rows, block.y));

    ComputeVertexMap_kernel<<<grid, block>>>(depth_inv, vmap, invfx, invfy, cx, cy, cut_off);
}

__global__ void ComputeNormalMap_kernel(const cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                        cv::cuda::PtrStep<Eigen::Vector4f> nmap)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= vmap.cols || y >= vmap.rows)
        return;

    nmap.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1.f);

    if (x <= 1 || y <= 1 || x >= vmap.cols - 1 || y >= vmap.rows - 1)
    {
        return;
    }

    Eigen::Vector4f v00 = vmap.ptr(y)[x - 1];
    Eigen::Vector4f v01 = vmap.ptr(y)[x + 1];
    Eigen::Vector4f v10 = vmap.ptr(y - 1)[x];
    Eigen::Vector4f v11 = vmap.ptr(y + 1)[x];

    if (v00(3) > 0 && v01(3) > 0 && v10(3) > 0 && v11(3) > 0)
    {
        nmap.ptr(y)[x].head<3>() = (v11 - v10).head<3>().cross((v01 - v00).head<3>()).normalized();
        nmap.ptr(y)[x](3) = 1.f;
    }
    else
    {
        nmap.ptr(y)[x](3) = -1.f;
    }
}

void ComputeNormalMap(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat nmap)
{
    if (nmap.empty())
        nmap.create(vmap.size(), CV_32FC4);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

    ComputeNormalMap_kernel<<<grid, block>>>(vmap, nmap);
}

__global__ void VMapToDepth_kernel(const cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                   cv::cuda::PtrStep<float> depth)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= vmap.cols || y >= vmap.rows)
        return;

    Eigen::Vector4f v = vmap.ptr(y)[x];
    if (v(3) > 0)
        depth.ptr(y)[x] = v(2);
    else
        depth.ptr(y)[x] = 0;
}

void VMapToDepth(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &depth)
{
    if (depth.empty())
        depth.create(vmap.size(), CV_32FC1);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

    VMapToDepth_kernel<<<grid, block>>>(vmap, depth);
}

__global__ void ComputeIntegralImageX_kernel(const cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                             cv::cuda::PtrStep<Eigen::Vector3f> intImg)
{
    int y = blockIdx.x;
    int x = threadIdx.x;

    // printf("x: %d, y: %d\n", x, y);

    __shared__ float xx[1024];
    __shared__ float yy[1024];
    __shared__ float zz[1024];

    if (threadIdx.x == 0)
    {
        memset(xx, 0, sizeof(float) * 1024);
        memset(yy, 0, sizeof(float) * 1024);
        memset(zz, 0, sizeof(float) * 1024);
    }

    __syncthreads();

    Eigen::Vector3f V(0, 0, 0);
    if (x < vmap.cols)
    {
        if (vmap.ptr(y)[x](3) > 0)
            V = vmap.ptr(y)[x].head<3>();
    }

    xx[x] = V(0);
    yy[x] = V(1);
    zz[x] = V(2);

    __syncthreads();

    int s1, s2;

    // Up sweep (reduce) phase
    for (s1 = 1, s2 = 1; s1 < 1024; s1 <<= 1)
    {
        s2 |= s1;
        if ((threadIdx.x & s2) == s2)
        {
            xx[threadIdx.x] += xx[threadIdx.x - s1];
            yy[threadIdx.x] += yy[threadIdx.x - s1];
            zz[threadIdx.x] += zz[threadIdx.x - s1];
        }

        __syncthreads();
    }

    // Down sweeping phase
    for (s1 >>= 2, s2 >>= 1; s1 >= 1; s1 >>= 1, s2 >>= 1)
    {
        if (threadIdx.x != 1023 && (threadIdx.x & s2) == s2)
        {
            xx[threadIdx.x + s1] += xx[threadIdx.x];
            yy[threadIdx.x + s1] += yy[threadIdx.x];
            zz[threadIdx.x + s1] += zz[threadIdx.x];
        }

        __syncthreads();
    }

    if (x < vmap.cols)
        intImg.ptr(y)[x] = Eigen::Vector3f(xx[x], yy[x], zz[x]);
}

__global__ void ComputeIntegralImageY_kernel(const cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                             cv::cuda::PtrStep<Eigen::Vector3f> intImg)
{
    int x = blockIdx.x;
    int y = threadIdx.x;

    // printf("x: %d, y: %d\n", x, y);

    __shared__ float xx[1024];
    __shared__ float yy[1024];
    __shared__ float zz[1024];

    if (threadIdx.x == 0)
    {
        memset(xx, 0, sizeof(float) * 1024);
        memset(yy, 0, sizeof(float) * 1024);
        memset(zz, 0, sizeof(float) * 1024);
    }

    __syncthreads();

    Eigen::Vector3f V(0, 0, 0);
    if (y < vmap.rows)
    {
        if (vmap.ptr(y)[x](3) > 0)
            V = vmap.ptr(y)[x].head<3>();
    }

    xx[threadIdx.x] = V(0);
    yy[threadIdx.x] = V(1);
    zz[threadIdx.x] = V(2);

    __syncthreads();

    int s1, s2;

    // Up sweep (reduce) phase
    for (s1 = 1, s2 = 1; s1 < 1024; s1 <<= 1)
    {
        s2 |= s1;
        if ((threadIdx.x & s2) == s2)
        {
            xx[threadIdx.x] += xx[threadIdx.x - s1];
            yy[threadIdx.x] += yy[threadIdx.x - s1];
            zz[threadIdx.x] += zz[threadIdx.x - s1];
        }

        __syncthreads();
    }

    // Down sweeping phase
    for (s1 >>= 2, s2 >>= 1; s1 >= 1; s1 >>= 1, s2 >>= 1)
    {
        if (threadIdx.x != 1023 && (threadIdx.x & s2) == s2)
        {
            xx[threadIdx.x + s1] += xx[threadIdx.x];
            yy[threadIdx.x + s1] += yy[threadIdx.x];
            zz[threadIdx.x + s1] += zz[threadIdx.x];
        }

        __syncthreads();
    }

    if (threadIdx.x < vmap.rows)
        intImg.ptr(y)[x] = Eigen::Vector3f(xx[threadIdx.x], yy[threadIdx.x], zz[threadIdx.x]);
}

void ComputeIntegralImage(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &IntImg)
{
    dim3 gridX(vmap.rows);
    ComputeIntegralImageX_kernel<<<gridX, 1024>>>(vmap, IntImg);

    dim3 gridY(vmap.cols);
    ComputeIntegralImageY_kernel<<<gridY, 1024>>>(vmap, IntImg);
}

// template <int R = 5>
// __global__ void ComputeNormalAndMeanCurvature_kernel(const cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
//                                                      cv::cuda::PtrStep<Eigen::Vector4f> nmap,
//                                                      cv::cuda::PtrStep<float> curvature)
// {
//     int x = blockDim.x * blockIdx.x + threadIdx.x;
//     int y = blockDim.y * blockIdx.y + threadIdx.y;
//     if (x >= vmap.cols || y >= vmap.rows)
//         return;

//     int r = R / 2;
//     if (x < r || y < r || x >= vmap.cols - r || y >= vmap.rows - r)
//     {
//         nmap.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1.f);
//         curvature.ptr(y)[x] = -1;
//         return;
//     }

//     Eigen::Vector3f centroid;
//     centroid.setZero();
//     int vcount;
// #pragma unroll
//     for (int i = x - r; i <= x + r; ++i)
// #pragma unroll
//         for (int j = y - r; j <= y + r; ++j)
//         {
//             if (vmap.ptr(j)[i](3) > 0)
//             {
//                 vcount++;
//                 centroid += vmap.ptr(j)[i].head<3>();
//             }
//         }

//     if (vcount == 0)
//     {
//         nmap.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1);
//         curvature.ptr(y)[x] = -1;
//         return;
//     }

//     centroid /= vcount;
//     Eigen::Matrix<float, 3, 3> A;
//     A.setZero();
// #pragma unroll
//     for (int i = x - r; i < x + r; ++i)
// #pragma unroll
//         for (int j = y - r; j < y + r; ++j)
//         {
//             if (vmap.ptr(j)[i](3) > 0)
//             {
//                 Eigen::Vector3f v = vmap.ptr(j)[i].head<3>() - centroid;
//                 A += v * v.transpose();
//             }
//         }

//     Eigen::Matrix<float, 3, 3> U, S, V;
//     svd(A, U, S, V);
//     float s1 = S(0, 0), s2 = S(1, 1), s3 = S(2, 2);

//     if (s1 == 0 || s2 == 0 || s3 == 0)
//     {
//         // nmap.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1);
//         curvature.ptr(y)[x] = -1;
//         return;
//     }

//     // nmap.ptr(y)[x].head<3>() = V.topRows<1>();
//     // nmap.ptr(y)[x](3) = 1.f;
//     curvature.ptr(y)[x] = s1 / (s1 + s2 + s3);

//     // printf("%f, %f, %f\n", s1, s2, s3);
// }

// void ComputeNormalAndMeanCurvature(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &nmap, cv::cuda::GpuMat &curvature)
// {
//     if (nmap.empty())
//         nmap.create(vmap.size(), CV_32FC4);
//     if (curvature.empty())
//         curvature.create(vmap.size(), CV_32FC1);

//     // cv::cuda::GpuMat IntImg(vmap.size(), CV_32FC3);
//     // ComputeIntegralImage(vmap, IntImg);

//     dim3 block(8, 8);
//     dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));
//     ComputeNormalAndMeanCurvature_kernel<<<grid, block>>>(vmap, nmap, curvature);
//     SafeCall(cudaDeviceSynchronize());

//     // if (nmap.cols == 640)
//     // {
//     //     cv::Mat out(nmap);
//     //     cv::imshow("nmap", out);
//     //     cv::waitKey(0);
//     // }
// }

__global__ void ComputeCurvature_kernel(const cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                        const cv::cuda::PtrStep<Eigen::Vector4f> nmap,
                                        cv::cuda::PtrStep<float> curvature)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= vmap.cols || y >= vmap.rows)
        return;

    if (x == 0 || y == 0 || x == vmap.cols - 1 || y == vmap.rows - 1)
    {
        curvature.ptr(y)[x] = 0;
        return;
    }

    Eigen::Vector4f v00, v01, v10, v11;
    Eigen::Vector4f v = vmap.ptr(y)[x];
    Eigen::Vector4f n = nmap.ptr(y)[x];
    v00 = vmap.ptr(y)[x + 1];
    v01 = vmap.ptr(y)[x - 1];
    v10 = vmap.ptr(y - 1)[x];
    v11 = vmap.ptr(y + 1)[x];

    if (n(3) < 0 || v(3) < 0 || v00(3) < 0 || v01(3) < 0 || v10(3) < 0 || v11(3) < 0)
    {
        curvature.ptr(y)[x] = 0;
        return;
    }

    float dx = (v00.head<3>() - v.head<3>()).dot(n.head<3>()) - (v.head<3>() - v01.head<3>()).dot(n.head<3>());
    float dy = (v11.head<3>() - v.head<3>()).dot(n.head<3>()) - (v.head<3>() - v10.head<3>()).dot(n.head<3>());
    float c = (dx + dy) * 0.5f;
    curvature.ptr(y)[x] = fabs(c) > 0.003f ? 255 : 0;
}

__global__ void DilateKernel(const cv::cuda::PtrStepSz<float> src, cv::cuda::PtrStep<float> dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= src.rows || x >= src.cols)
        return;

    float r = src.ptr(y)[x];
    if (x < 1 || x >= src.cols - 1 || y < 1 || y >= src.rows - 1)
        dst.ptr(y)[x] = r;

    r = fmax(src.ptr(y - 1)[x - 1], r);
    r = fmax(src.ptr(y - 1)[x], r);
    r = fmax(src.ptr(y - 1)[x + 1], r);
    r = fmax(src.ptr(y)[x - 1], r);
    r = fmax(src.ptr(y)[x + 1], r);
    r = fmax(src.ptr(y + 1)[x - 1], r);
    r = fmax(src.ptr(y + 1)[x], r);
    r = fmax(src.ptr(y + 1)[x + 1], r);
    dst.ptr(y)[x] = r;
}

__global__ void ErodeKernel(const cv::cuda::PtrStepSz<float> src, cv::cuda::PtrStep<float> dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= src.rows || x >= src.cols)
        return;

    float r = src.ptr(y)[x];
    if (x < 1 || x >= src.rows - 1 || y < 1 || y >= src.cols - 1)
        dst.ptr(y)[x] = r;

    r = fmin(src.ptr(y - 1)[x - 1], r);
    r = fmin(src.ptr(y - 1)[x], r);
    r = fmin(src.ptr(y - 1)[x + 1], r);
    r = fmin(src.ptr(y)[x - 1], r);
    r = fmin(src.ptr(y)[x + 1], r);
    r = fmin(src.ptr(y + 1)[x - 1], r);
    r = fmin(src.ptr(y + 1)[x], r);
    r = fmin(src.ptr(y + 1)[x + 1], r);
    dst.ptr(y)[x] = r;
}

void morphGeometricSegmentationMap(const cv::cuda::GpuMat src,
                                   const cv::cuda::GpuMat dst2)
{

    cv::cuda::GpuMat dst(src.size(), CV_32FC1);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = cv::divUp(src.cols, block.x);
    grid.y = cv::divUp(src.rows, block.y);
    DilateKernel<<<grid, block>>>(src, dst);
    ErodeKernel<<<grid, block>>>(dst, src);
    DilateKernel<<<grid, block>>>(src, dst);
    ErodeKernel<<<grid, block>>>(dst, src);
    DilateKernel<<<grid, block>>>(src, dst);
    ErodeKernel<<<grid, block>>>(dst, src);

    cv::Mat out(dst);
    cv::imshow("out", out);
    cv::waitKey(0);
}

__device__ float getConcavityTerm(const cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                  const cv::cuda::PtrStepSz<Eigen::Vector4f> nmap,
                                  const Eigen::Vector3f &v,
                                  const Eigen::Vector3f &n,
                                  int x_n, int y_n)
{
    const Eigen::Vector3f v_n = vmap.ptr(y_n)[x_n].head<3>();
    const Eigen::Vector3f n_n = nmap.ptr(y_n)[x_n].head<3>();
    if ((v_n - v).dot(n) < 0)
        return 0;
    return 1 - n_n.dot(n);
}

__device__ float getDistanceTerm(const cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                 const Eigen::Vector3f &v,
                                 const Eigen::Vector3f &n,
                                 int x_n, int y_n)
{
    const Eigen::Vector3f v_n = vmap.ptr(y_n)[x_n].head<3>();
    Eigen::Vector3f d = v_n - v;
    return fabs(d.dot(n));
}

__global__ void computeGeometricSegmentation_Kernel(const cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                                    const cv::cuda::PtrStepSz<Eigen::Vector4f> nmap,
                                                    cv::cuda::PtrStepSz<float> output,
                                                    float wD, float wC)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= vmap.rows || x >= vmap.cols)
        return;

    const int radius = 1;
    if (x < radius || x >= vmap.cols - radius || y < radius || y >= vmap.rows - radius)
    {
        output.ptr(y)[x] = 1.0f;
        return;
    }

    const Eigen::Vector3f v = vmap.ptr(y)[x].head<3>();
    const Eigen::Vector3f n = nmap.ptr(y)[x].head<3>();
    if (vmap.ptr(y)[x](3) <= 0.0f || nmap.ptr(y)[x](3) < 0)
    {
        output.ptr(y)[x] = 1.0f;
        return;
    }

    float c = 0.0f;
    c = fmax(getConcavityTerm(vmap, nmap, v, n, x - radius, y - radius), c);
    c = fmax(getConcavityTerm(vmap, nmap, v, n, x, y - radius), c);
    c = fmax(getConcavityTerm(vmap, nmap, v, n, x + radius, y - radius), c);
    c = fmax(getConcavityTerm(vmap, nmap, v, n, x - radius, y), c);
    c = fmax(getConcavityTerm(vmap, nmap, v, n, x + radius, y), c);
    c = fmax(getConcavityTerm(vmap, nmap, v, n, x - radius, y + radius), c);
    c = fmax(getConcavityTerm(vmap, nmap, v, n, x, y + radius), c);
    c = fmax(getConcavityTerm(vmap, nmap, v, n, x + radius, y + radius), c);
    c = fmax(c, 0.0f);
    c *= wC;

    float d = 0.0f;
    d = fmax(getDistanceTerm(vmap, v, n, x - radius, y - radius), d);
    d = fmax(getDistanceTerm(vmap, v, n, x, y - radius), d);
    d = fmax(getDistanceTerm(vmap, v, n, x + radius, y - radius), d);
    d = fmax(getDistanceTerm(vmap, v, n, x - radius, y), d);
    d = fmax(getDistanceTerm(vmap, v, n, x + radius, y), d);
    d = fmax(getDistanceTerm(vmap, v, n, x - radius, y + radius), d);
    d = fmax(getDistanceTerm(vmap, v, n, x, y + radius), d);
    d = fmax(getDistanceTerm(vmap, v, n, x + radius, y + radius), d);
    d *= wD;

    float edgeness = fmax(c, d);
    output.ptr(y)[x] = fmin(1.0f, edgeness);
}

void ComputeCurvature(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat &nmap, cv::cuda::GpuMat &curvature)
{
    if (curvature.empty())
        curvature.create(vmap.size(), CV_32FC1);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

    ComputeCurvature_kernel<<<grid, block>>>(vmap, nmap, curvature);
    // computeGeometricSegmentation_Kernel<<<grid, block>>>(vmap, nmap, curvature, 1, 1);
    // morphGeometricSegmentationMap(curvature, curvature);

    // if (vmap.cols == 640)
    // {
    //     cv::Mat out(curvature);
    //     cv::Mat out2;
    //     out.convertTo(out2, CV_8UC1);
    //     cv::imshow("nmap", out2);
    //     cv::waitKey(0);
    // }
}

__global__ void PyrDownImage_kernel(const cv::cuda::PtrStep<float> src, cv::cuda::PtrStepSz<float> dst)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= dst.cols || y >= dst.rows)
        return;

    dst.ptr(y)[x] = 0.25 * (src.ptr(y * 2)[x * 2] + src.ptr(y * 2)[x * 2 + 1] + src.ptr(y * 2 + 1)[x * 2] + src.ptr(y * 2 + 1)[x * 2 + 1]);
}

void PyrDownImage(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
{
    if (dst.empty())
        dst.create(src.rows / 2, src.cols / 2, CV_32FC1);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(dst.cols, block.x), cv::divUp(dst.rows, block.y));

    PyrDownImage_kernel<<<grid, block>>>(src, dst);
}

__global__ void PyrDownVec4f_kernel(const cv::cuda::PtrStep<Eigen::Vector4f> src, cv::cuda::PtrStepSz<Eigen::Vector4f> dst)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= dst.cols || y >= dst.rows)
        return;

    Eigen::Vector4f v;
    Eigen::Vector3f vsum(0, 0, 0);
    int vcount = 0;

    v = src.ptr(y * 2)[x * 2];
    if (v(3) > 0)
    {
        vcount++;
        vsum += v.head<3>();
    }

    v = src.ptr(y * 2)[x * 2 + 1];
    if (v(3) > 0)
    {
        vcount++;
        vsum += v.head<3>();
    }

    v = src.ptr(y * 2 + 1)[x * 2];
    if (v(3) > 0)
    {
        vcount++;
        vsum += v.head<3>();
    }

    v = src.ptr(y * 2 + 1)[x * 2 + 1];
    if (v(3) > 0)
    {
        vcount++;
        vsum += v.head<3>();
    }

    if (vcount == 0)
    {
        dst.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1);
    }
    else
    {
        v.head<3>() = vsum / vcount;
        v(3) = 1.f;
        dst.ptr(y)[x] = v;
    }
}

void PyrDownVec4f(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
{
    if (dst.empty())
        dst.create(src.rows / 2, src.cols / 2, CV_32FC1);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(dst.cols, block.x), cv::divUp(dst.rows, block.y));

    PyrDownVec4f_kernel<<<grid, block>>>(src, dst);
}

void computeMeanOpticalShift(cv::cuda::GpuMat srcImage, cv::cuda::GpuMat srcDepth, cv::cuda::GpuMat dstImage, const Sophus::SE3d &T, float &meanShift)
{
}