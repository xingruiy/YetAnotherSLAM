#include "ImageProc.h"
#include "CudaUtils.h"

__global__ void computeImageGradientCentralDiffKernel(
    cv::cuda::PtrStepSz<float> src,
    cv::cuda::PtrStep<float> gradientX,
    cv::cuda::PtrStep<float> gradientY)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= src.cols || y >= src.rows)
        return;

    int ym1 = max(0, y - 1);
    int yp1 = min(src.rows - 1, y + 1);
    int xm1 = max(0, x - 1);
    int xp1 = min(src.cols - 1, x + 1);

    gradientX.ptr(y)[x] = (src.ptr(y)[xp1] - src.ptr(y)[xm1]) * 0.5f;
    gradientY.ptr(y)[x] = (src.ptr(yp1)[x] - src.ptr(ym1)[x]) * 0.5f;
}

void computeImageGradientCentralDiff(cv::cuda::GpuMat image, cv::cuda::GpuMat &gx, cv::cuda::GpuMat &gy)
{
    if (gx.empty())
        gx.create(image.size(), CV_32FC1);
    if (gy.empty())
        gy.create(image.size(), CV_32FC1);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(image.cols, block.x), cv::divUp(image.rows, block.y));

    computeImageGradientCentralDiffKernel<<<grid, block>>>(image, gx, gy);
    // cudaCheckError();
}

__global__ void TransformReferencePointKernel(
    cv::cuda::PtrStepSz<float> depth,
    cv::cuda::PtrStep<Eigen::Vector4f> ptTransformed,
    Eigen::Matrix3f RKinv, Eigen::Vector3f t)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= depth.cols || y >= depth.rows)
        return;

    const float &zInv = depth.ptr(y)[x];
    if (zInv > FLT_EPSILON)
    {
        float z = 1.0 / zInv;
        Eigen::Vector3f pt = RKinv * Eigen::Vector3f(x, y, 1.0f) * z + t;
        ptTransformed.ptr(y)[x] = Eigen::Vector4f(pt(0), pt(1), pt(2), 1.0f);
    }
    else
        ptTransformed.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1.0f);
}

void TransformReferencePoint(cv::cuda::GpuMat depth, cv::cuda::GpuMat &vmap, const Eigen::Matrix3d &K, const Sophus::SE3d &T)
{
    if (vmap.empty())
        vmap.create(depth.size(), CV_32FC4);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(depth.cols, block.x), cv::divUp(depth.rows, block.y));

    Eigen::Matrix3d RKinv = T.matrix().topLeftCorner(3, 3) * K.inverse();
    Eigen::Vector3d t = T.matrix().topRightCorner(3, 1);

    TransformReferencePointKernel<<<grid, block>>>(depth, vmap, RKinv.cast<float>(), t.cast<float>());
    // cudaCheckError();
}

__device__ __forceinline__ Eigen::Matrix<uchar, 4, 1> renderPoint(
    const Eigen::Vector3f &point, const Eigen::Vector3f &normal,
    const Eigen::Vector3f &image, const Eigen::Vector3f &lightPos)
{
    Eigen::Vector3f colour(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
    if (!isnan(point(0)))
    {
        // ambient coeff
        const float Ka = 0.3f;
        // diffuse coeff
        const float Kd = 0.5f;
        // specular coeff
        const float Ks = 0.2f;
        // specular power
        const float n = 20.f;

        // ambient color
        const float Ax = image(0);
        // diffuse color
        const float Dx = image(1);
        // specular color
        const float Sx = image(2);
        // light color
        const float Lx = 1.f;

        Eigen::Vector3f L = (lightPos - point).normalized();
        Eigen::Vector3f V = (Eigen::Vector3f(0.f, 0.f, 0.f) - point).normalized();
        Eigen::Vector3f R = (2 * normal * (normal.dot(L)) - L).normalized();

        float Ix = Ax * Ka * Dx + Lx * Kd * Dx * fmax(0.f, (normal.dot(L))) + Lx * Ks * Sx * pow(fmax(0.f, (R.dot(V))), n);
        colour = Eigen::Vector3f(Ix, Ix, Ix);
    }

    return Eigen::Matrix<uchar, 4, 1>(
        static_cast<uchar>(__saturatef(colour(0)) * 255.f),
        static_cast<uchar>(__saturatef(colour(1)) * 255.f),
        static_cast<uchar>(__saturatef(colour(2)) * 255.f),
        255);
}

__global__ void renderSceneKernel(
    const cv::cuda::PtrStep<Eigen::Vector4f> vmap,
    const cv::cuda::PtrStep<Eigen::Vector4f> nmap,
    const Eigen::Vector3f lightPos,
    cv::cuda::PtrStepSz<Eigen::Matrix<uchar, 4, 1>> dst)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= dst.cols || y >= dst.rows)
        return;

    Eigen::Vector3f point = vmap.ptr(y)[x].head<3>();
    Eigen::Vector3f normal = nmap.ptr(y)[x].head<3>();
    Eigen::Vector3f pixel(1.f, 1.f, 1.f);

    dst.ptr(y)[x] = renderPoint(point, normal, pixel, lightPos);
}

void renderScene(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, cv::cuda::GpuMat &image)
{
    if (image.empty())
        image.create(vmap.size(), CV_8UC4);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

    renderSceneKernel<<<grid, block>>>(vmap, nmap, Eigen::Vector3f(5, 5, 5), image);
}

__global__ void computeNormalKernel(cv::cuda::PtrStepSz<Eigen::Vector4f> vmap, cv::cuda::PtrStep<Eigen::Vector4f> nmap)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= vmap.cols - 1 || y >= vmap.rows - 1)
        return;

    int x10 = max(x - 1, 0);
    int x01 = min(x + 1, vmap.cols);
    int y10 = max(y - 1, 0);
    int y01 = min(y + 1, vmap.rows);

    Eigen::Vector3f v00 = vmap.ptr(y)[x10].head<3>();
    Eigen::Vector3f v01 = vmap.ptr(y)[x01].head<3>();
    Eigen::Vector3f v10 = vmap.ptr(y10)[x].head<3>();
    Eigen::Vector3f v11 = vmap.ptr(y01)[x].head<3>();

    nmap.ptr(y)[x].head<3>() = ((v01 - v00).cross(v11 - v10)).normalized();
    nmap.ptr(y)[x](3) = 1.f;
}

void computeNormal(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &nmap)
{
    if (nmap.empty())
        nmap.create(vmap.size(), vmap.type());

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

    computeNormalKernel<<<grid, block>>>(vmap, nmap);
    // cudaCheckError();
}

__global__ void convertDepthToInvDepthKernel(
    cv::cuda::PtrStep<float> depth,
    cv::cuda::PtrStepSz<float> invDepth)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= invDepth.cols - 1 || y >= invDepth.rows - 1)
        return;

    const float z = depth.ptr(y)[x];
    if (z == z && z > FLT_EPSILON)
    {
        invDepth.ptr(y)[x] = 1.0 / z;
    }
    else
    {
        invDepth.ptr(y)[x] = 0;
    }
}

void convertDepthToInvDepth(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &invDepth)
{
    if (invDepth.empty())
        invDepth.create(depth.size(), depth.type());

    dim3 block(8, 8);
    dim3 grid(cv::divUp(depth.cols, block.x), cv::divUp(depth.rows, block.y));

    convertDepthToInvDepthKernel<<<grid, block>>>(depth, invDepth);
}

__global__ void convertVMapToInvDepthKernel(
    cv::cuda::PtrStep<Eigen::Vector4f> vmap,
    cv::cuda::PtrStepSz<float> invDepth)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= invDepth.cols || y >= invDepth.rows)
        return;

    const auto pt = vmap.ptr(y)[x];
    if (pt(3) > 0)
    {
        invDepth.ptr(y)[x] = 1.0 / pt(2);
    }
    // else
    // {
    //     invDepth.ptr(y)[x] = 0;
    // }
}

void convertVMapToInvDepth(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &invDepth)
{
    if (invDepth.empty())
        invDepth.create(vmap.size(), CV_32FC1);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

    convertVMapToInvDepthKernel<<<grid, block>>>(vmap, invDepth);
    // cudaCheckError();
}

__global__ void pyrdownInvDepthKernel(
    cv::cuda::PtrStep<float> src,
    cv::cuda::PtrStepSz<float> dst)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= dst.cols - 1 || y >= dst.rows - 1)
        return;

    dst.ptr(y)[x] = src.ptr(2 * y)[2 * x];
}

void pyrdownInvDepth(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
{
    if (dst.empty())
        dst.create(src.size(), CV_32FC1);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(src.cols, block.x), cv::divUp(src.rows, block.y));

    pyrdownInvDepthKernel<<<grid, block>>>(src, dst);
    // cudaCheckError();
}

__global__ void computeVMapKernel(
    const cv::cuda::PtrStep<float> depth,
    cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
    const float invfx, const float invfy,
    const float cx, const float cy)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= vmap.cols || y >= vmap.rows)
        return;

    const auto &z = depth.ptr(y)[x];
    Eigen::Vector4f v = Eigen::Vector4f(0, 0, 0, 0);
    if (z == z && z > FLT_EPSILON)
    {
        v(0) = (x - cx) * invfx * z;
        v(1) = (y - cy) * invfy * z;
        v(2) = z;
        v(3) = 1.0;
    }
    else
    {
        v(0) = nanf("error");
    }

    vmap.ptr(y)[x] = v;
}

void computeVMap(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &vmap, const Eigen::Matrix3d &K)
{
    if (vmap.empty())
        vmap.create(depth.rows, depth.cols, CV_32FC4);

    const float invfx = 1.0 / K(0, 0);
    const float invfy = 1.0 / K(1, 1);
    const float cx = K(0, 2);
    const float cy = K(1, 2);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));
    computeVMapKernel<<<grid, block>>>(depth, vmap, invfx, invfy, cx, cy);
}