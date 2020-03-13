#include "LocalBundler.h"
#include "CudaUtils.h"
#include "ImageProc.h"

LocalBundler::LocalBundler(int w, int h, const Eigen::Matrix3f &K) : calib(K), width(w), height(h)
{
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
}

__global__ void CheckProjections_kernel(PointShell *points, int N,
                                        Sophus::SE3f Tinv, int idx,
                                        int *num_pts, int w, int h,
                                        float fx, float fy,
                                        float cx, float cy)
{
    float sum[1] = {0};

    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
        if (P.state == OK || P.state == Unchecked)
        {
            const float z = 1.0 / P.idepth;
            Eigen::Vector3f pt = Tinv * Eigen::Vector3f(z * (P.x - cx) / fx, z * (P.y - cy) / fy, z);
            const float u = fx * pt(0) / pt(2) + cx;
            const float v = fy * pt(1) / pt(2) + cy;
            if (u < 1 || v < 1 || u >= w - 1 || v >= h - 1)
            {
                P.state = OOB;
            }
            else
            {
                P.state = OK;
                RawResidual R(P.res[idx]);
                R.uv = Eigen::Vector2f(u, v);
                P.numResiduals++;
            }
        }
    }
}

void LocalBundler::CheckProjections(FrameShell &F)
{
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

void LocalBundler::BundleAdjust(const int maxIter)
{
}

void LocalBundler::CreateNewPoints()
{
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