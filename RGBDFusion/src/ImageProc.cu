#include "ImageProc.h"
#include "CudaUtils.h"

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

__device__ __forceinline__ Eigen::Matrix<uchar, 4, 1> RenderPoint(const Eigen::Vector3f &point,
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

    return Eigen::Matrix<uchar, 4, 1>(static_cast<uchar>(__saturatef(colour(0)) * 255.f),
                                      static_cast<uchar>(__saturatef(colour(1)) * 255.f),
                                      static_cast<uchar>(__saturatef(colour(2)) * 255.f),
                                      255);
}

__global__ void RenderScene_kernel(const cv::cuda::PtrStep<Eigen::Vector4f> vmap,
                                   const cv::cuda::PtrStep<Eigen::Vector4f> nmap,
                                   const Eigen::Vector3f lightPos,
                                   cv::cuda::PtrStepSz<Eigen::Matrix<uchar, 4, 1>> dst)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= dst.cols - 1 || y >= dst.rows - 1)
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

    RenderScene_kernel<<<grid, block>>>(vmap, nmap, Eigen::Vector3f(-5, -5, -5), image);

    SafeCall(cudaDeviceSynchronize());
    SafeCall(cudaGetLastError());
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
        vmap.ptr(y)[x](3) = -1.f;
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

    if (x == 0 || y == 0 || x == vmap.cols - 1 || y == vmap.rows - 1)
    {
        nmap.ptr(y)[x](3) = -1.f;
        return;
    }

    Eigen::Vector4f v00 = vmap.ptr(y)[x];
    Eigen::Vector4f v01 = vmap.ptr(y)[x + 1];
    Eigen::Vector4f v10 = vmap.ptr(y + 1)[x];

    if (v00(3) > 0 && v01(3) > 0 && v10(3) > 0)
    {
        nmap.ptr(y)[x].head<3>() = (v10 - v00).head<3>().cross((v01 - v00).head<3>()).normalized();
        nmap.ptr(y)[x](3) = 1.f;
    }
    else
    {
        nmap.ptr(y)[x](3) = -1.f;
    }
}

void ComputeNormalMap(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &nmap)
{
    if (nmap.empty())
        nmap.create(vmap.size(), CV_32FC4);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

    ComputeNormalMap_kernel<<<grid, block>>>(vmap, nmap);

    SafeCall(cudaDeviceSynchronize());
    SafeCall(cudaGetLastError());
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