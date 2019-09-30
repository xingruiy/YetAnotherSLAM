#include "denseTracker/cudaImageProc.h"
#include "utils/numType.h"
#include "utils/cudaUtils.h"

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

void computeImageGradientCentralDiff(GMat image, GMat &gx, GMat &gy)
{
    if (gx.empty())
        gx.create(image.size(), CV_32FC1);
    if (gy.empty())
        gy.create(image.size(), CV_32FC1);

    dim3 block(8, 8);
    dim3 grid = getGridConfiguration2D(block, image.cols, image.rows);
    computeImageGradientCentralDiffKernel<<<grid, block>>>(image, gx, gy);
    // cudaCheckError();
}

__global__ void transformReferencePointKernel(
    cv::cuda::PtrStepSz<float> depth,
    cv::cuda::PtrStep<Vec4f> ptTransformed,
    Mat33f RKinv, Vec3f t)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= depth.cols || y >= depth.rows)
        return;

    const float &zInv = depth.ptr(y)[x];
    if (zInv > FLT_EPSILON)
    {
        float z = 1.0 / zInv;
        Vec3f pt = RKinv * Vec3f(x, y, 1.0f) * z + t;
        ptTransformed.ptr(y)[x] = Vec4f(pt(0), pt(1), pt(2), 1.0f);
    }
    else
        ptTransformed.ptr(y)[x] = Vec4f(0, 0, 0, -1.0f);
}

void transformReferencePoint(GMat depth, GMat &vmap, const Mat33d &K, const SE3 &T)
{
    if (vmap.empty())
        vmap.create(depth.size(), CV_32FC4);

    dim3 block(8, 8);
    dim3 grid = getGridConfiguration2D(block, depth.cols, depth.rows);

    Mat33d RKinv = T.matrix().topLeftCorner(3, 3) * K.inverse();
    Vec3d t = T.matrix().topRightCorner(3, 1);

    transformReferencePointKernel<<<grid, block>>>(depth, vmap, RKinv.cast<float>(), t.cast<float>());
    // cudaCheckError();
}

__device__ __forceinline__ Vec4b renderPoint(
    const Vec3f &point, const Vec3f &normal,
    const Vec3f &image, const Vec3f &lightPos)
{
    Vec3f colour(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
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

        Vec3f L = (lightPos - point).normalized();
        Vec3f V = (Vec3f(0.f, 0.f, 0.f) - point).normalized();
        Vec3f R = (2 * normal * (normal.dot(L)) - L).normalized();

        float Ix = Ax * Ka * Dx + Lx * Kd * Dx * fmax(0.f, (normal.dot(L))) + Lx * Ks * Sx * pow(fmax(0.f, (R.dot(V))), n);
        colour = Vec3f(Ix, Ix, Ix);
    }

    return Vec4b(
        static_cast<uchar>(__saturatef(colour(0)) * 255.f),
        static_cast<uchar>(__saturatef(colour(1)) * 255.f),
        static_cast<uchar>(__saturatef(colour(2)) * 255.f),
        255);
}

__global__ void renderSceneKernel(
    const cv::cuda::PtrStep<Vec4f> vmap,
    const cv::cuda::PtrStep<Vec4f> nmap,
    const Vec3f lightPos,
    cv::cuda::PtrStepSz<Vec4b> dst)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= dst.cols || y >= dst.rows)
        return;

    Vec3f point = vmap.ptr(y)[x].head<3>();
    Vec3f normal = nmap.ptr(y)[x].head<3>();
    Vec3f pixel(1.f, 1.f, 1.f);

    dst.ptr(y)[x] = renderPoint(point, normal, pixel, lightPos);
}

void renderScene(const GMat vmap, const GMat nmap, GMat &image)
{
    if (image.empty())
        image.create(vmap.size(), CV_8UC4);

    dim3 block(8, 8);
    dim3 grid = getGridConfiguration2D(block, vmap.cols, vmap.rows);

    renderSceneKernel<<<grid, block>>>(vmap, nmap, Vec3f(5, 5, 5), image);
}

__global__ void computeNormalKernel(cv::cuda::PtrStepSz<Vec4f> vmap, cv::cuda::PtrStep<Vec4f> nmap)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= vmap.cols - 1 || y >= vmap.rows - 1)
        return;

    int x10 = max(x - 1, 0);
    int x01 = min(x + 1, vmap.cols);
    int y10 = max(y - 1, 0);
    int y01 = min(y + 1, vmap.rows);

    Vec3f v00 = vmap.ptr(y)[x10].head<3>();
    Vec3f v01 = vmap.ptr(y)[x01].head<3>();
    Vec3f v10 = vmap.ptr(y10)[x].head<3>();
    Vec3f v11 = vmap.ptr(y01)[x].head<3>();

    nmap.ptr(y)[x].head<3>() = ((v01 - v00).cross(v11 - v10)).normalized();
    nmap.ptr(y)[x](3) = 1.f;
}

void computeNormal(const GMat vmap, GMat &nmap)
{
    if (nmap.empty())
        nmap.create(vmap.size(), vmap.type());

    dim3 block(8, 8);
    dim3 grid = getGridConfiguration2D(block, vmap.cols, vmap.rows);

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

void convertDepthToInvDepth(const GMat depth, GMat &invDepth)
{
    if (invDepth.empty())
        invDepth.create(depth.size(), depth.type());

    dim3 block(8, 8);
    dim3 grid = getGridConfiguration2D(block, depth.cols, depth.rows);

    convertDepthToInvDepthKernel<<<grid, block>>>(depth, invDepth);
    // cudaCheckError();
}

__global__ void convertVMapToInvDepthKernel(
    cv::cuda::PtrStep<Vec4f> vmap,
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

void convertVMapToInvDepth(const GMat vmap, GMat &invDepth)
{
    if (invDepth.empty())
        invDepth.create(vmap.size(), CV_32FC1);

    dim3 block(8, 8);
    dim3 grid = getGridConfiguration2D(block, vmap.cols, vmap.rows);

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

void pyrdownInvDepth(const GMat src, GMat &dst)
{
    if (dst.empty())
        dst.create(src.size(), CV_32FC1);

    dim3 block(8, 8);
    dim3 grid = getGridConfiguration2D(block, src.cols, src.rows);

    pyrdownInvDepthKernel<<<grid, block>>>(src, dst);
    // cudaCheckError();
}