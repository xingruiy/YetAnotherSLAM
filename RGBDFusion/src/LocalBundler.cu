#include "LocalBundler.h"
#include "CudaUtils.h"
#include "ImageProc.h"
#include "TrackingUtils.h"

LocalBundler::LocalBundler(int w, int h, const Eigen::Matrix3f &K) : calib(K), width(w), height(h)
{
    int DOF = NUM_KF - 1;
    Hcc.create(1, DOF * DOF * 21, CV_32FC1);
    bcc.create(1, DOF * 6, CV_32FC1);

    Reset();
}

LocalBundler::~LocalBundler()
{
}

void LocalBundler::Reset()
{
}

void LocalBundler::AddKeyFrame(const cv::Mat depth,
                               const cv::Mat image,
                               const Sophus::SE3d &Tcw)
{
    FrameShell F;
    F.image = cv::cuda::GpuMat(image);
    ComputeImageGradientCentralDifference(F.image, F.dIdx, F.dIdy);

    F.depth = cv::cuda::GpuMat(depth);
    F.Tcw = Tcw;
    frames.push_back(F);
    assert(frames.size() <= NUM_KF);
}

__global__ void LineariseAll_kernel(PointShell *points, int N,
                                    Sophus::SE3f *poseMatrix,
                                    int w, int h,
                                    float fx, float fy,
                                    float cx, float cy)
{
    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
        if (P.numResiduals != 0)
        {
            const int &hostIdx = P.frameIdx;
            for (int i = 0; i < NUM_KF; ++i)
            {
                if (!P.res[i].active || i == P.frameIdx)
                    continue;

                RawResidual &res(P.res[i]);
                const int targetIdx = res.targetIdx;
                const Sophus::SE3f &T(poseMatrix[hostIdx * NUM_KF + targetIdx]);
                const float z = 1.0 / P.idepth;
                Eigen::Vector3f pt = T * Eigen::Vector3f(z * (P.x - cx) / fx, z * (P.y - cy) / fy, z);
                const float u = fx * pt(0) / pt(2) + cx;
                const float v = fy * pt(1) / pt(2) + cy;
                if (u < 1 || v < 1 || u >= w - 1 || v >= h - 1)
                {
                    res.state = OOB;
                }
                else
                {
                    res.state = OK;
                    const float invz = 1.0 / pt(2);
                    const float invz2 = invz * invz;
                    res.Jpdxi(0, 0) = fx * invz;
                    res.Jpdxi(0, 1) = 0;
                    res.Jpdxi(0, 2) = -fx * pt(0) * pt(1) * invz2;
                    res.Jpdxi(0, 3) = fx + fx * pt(0) * pt(0) * invz2;
                    res.Jpdxi(0, 4) = fx + fx * pt(0) * pt(0) * invz2;
                    res.Jpdxi(0, 5) = -fx * pt(0) * invz;
                    res.Jpdxi(1, 0) = 0;
                    res.Jpdxi(1, 1) = fy * invz;
                    res.Jpdxi(1, 2) = -fy * pt(1) * invz2;
                    res.Jpdxi(1, 3) = -fy - fy * pt(1) * pt(1) * invz2;
                    res.Jpdxi(1, 4) = fy * pt(0) * pt(1) * invz2;
                    res.Jpdxi(1, 5) = fy * pt(0) * invz;
                    res.uv = Eigen::Vector2f(u, v);
                }
            }
        }
    }
}

void LocalBundler::Marginalization()
{
    for (auto vit = frames.begin(), vend = frames.end(); vit != vend; ++vit)
    {
        bool bMarginalized = false;

        if (bMarginalized)
            vit = frames.erase(vit);
    }
}

__device__ __forceinline__ float GetHuberWeight(float r, const float huberTh)
{
    return r > huberTh ? huberTh : r;
}

__global__ void BuildCameraSystem_kernel(PointShell *points, int N,
                                         int targetIdx,
                                         cv::cuda::PtrStepSz<float> im,
                                         cv::cuda::PtrStep<float> gx,
                                         cv::cuda::PtrStep<float> gy,
                                         cv::cuda::PtrStep<float> out)
{
    float sum[21];
    memset(&sum[0], 0, sizeof(float) * 21);

    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
        if (P.numResiduals != 0)
        {
            for (int i = 0; i < NUM_KF; ++i)
            {
                RawResidual &res(P.res[i]);
                if (targetIdx != res.targetIdx || !res.active)
                    continue;

                const float &u = res.uv(0);
                const float &v = res.uv(1);

                float i = interpolateBiLinear(im, u, v);
                float dx = interpolateBiLinear(gx, u, v);
                float dy = interpolateBiLinear(gy, u, v);
                if (isfinite(i) && isfinite(dx) && isfinite(dy))
                {
                    res.JIdp = Eigen::Vector2f(dx, dy);
                    res.hw = GetHuberWeight(res.r, 1.4);

                    Eigen::Matrix<float, 7, 1> row;
                    row(6) = res.r = P.intensity - i;
                    row.head<6>() = res.JIdp * res.Jpdxi;

                    int count = 0;
#pragma unroll
                    for (int k = 0; k < 7; ++k)
#pragma unroll
                        for (int j = k; j < 7; ++j)
                            sum[count++] = res.hw * row(k) * row(j);
                }
                else
                {
                }
            }
        }
    }

    BlockReduceSum<float, 21>(sum);

    if (threadIdx.x == 0)
#pragma unroll
        for (int k = 0; k < 21; ++k)
            out.ptr(blockIdx.x)[k] = sum[k];
}

void LocalBundler::BuildCameraSystem()
{
}

__global__ void BuildCameraSystem_kernel(PointShell *points, int N)
{
    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
        if (P.numResiduals != 0)
        {
            for (int i = 0; i < NUM_KF; ++i)
            {
                RawResidual &res(P.res[i]);
                P.Hs += res.hw * res.JIdp * res.Jpdd;
            }
        }
    }
}

__global__ void BuildOffDiagnolTerms_kernel(PointShell *points, int N,
                                            cv::cuda::PtrStep<float> out,
                                            int i, int j)
{
    float sum[21];
    memset(&sum, 0, sizeof(float) * 21);

    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
    }
}

void LocalBundler::BuildStructureSystem()
{
}

void LocalBundler::BundleAdjust(const int maxIter)
{
    for (int iter = 0; iter < maxIter; ++iter)
    {
        LineariseAll();
        BuildCameraSystem();
        BuildStructureSystem();
    }
}

void LocalBundler::UpdatePoseMatrix(FrameShell &F)
{
    Sophus::SE3d FTwc = F.Tcw.inverse();
    Sophus::SE3f poses[NUM_KF];

    for (int i = 0; i < frames.size(); ++i)
    {
        FrameShell &F2 = frames[i];
        poses[i] = (FTwc * F2.Tcw).cast<float>();
    }

    SafeCall(cudaMemcpy(posesMatrix_dev, &poses[0], sizeof(Sophus::SE3f) * NUM_KF, cudaMemcpyHostToDevice));
}

void LocalBundler::PopulateOccupancyGrid(FrameShell &F)
{
    if (F.OGrid.empty())
        F.OGrid.create(160, 210, CV_8SC1);

    for (auto vit = frames.begin(), vend = frames.end(); vit != vend; ++vit)
    {
        FrameShell &F2 = *vit;
        Sophus::SE3d Rt = F.Tcw.inverse() * F2.Tcw;
    }
}

void LocalBundler::LineariseAll()
{
}