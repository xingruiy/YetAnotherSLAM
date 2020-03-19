#include "LocalBundler.h"
#include "CudaUtils.h"
#include "ImageProc.h"
#include "TrackingUtils.h"

__constant__ Sophus::SE3f *constPoseMat;

__device__ Sophus::SE3f GetPoseFromMatrix(int hostIdx, int targetIdx)
{
    return constPoseMat[hostIdx * NUM_KF + targetIdx];
}

__device__ void SetPoseMatrix(const Sophus::SE3f &T, int hostIdx, int targetIdx)
{
    constPoseMat[hostIdx * NUM_KF + targetIdx] = T;
}

void UpdatePoseMatrix(std::array<Sophus::SE3d, NUM_KF> &poses)
{
    Sophus::SE3f poses_host[NUM_KF * NUM_KF];
    for (int i = 0; i < poses.size(); ++i)
        for (int j = 0; j < poses.size(); ++j)
            if (i != j)
                poses_host[i * NUM_KF + j] = (poses[j].inverse() * poses[i]).cast<float>();

    SafeCall(cudaMemcpyToSymbol(constPoseMat, poses_host, sizeof(Sophus::SE3f) * NUM_KF * NUM_KF));
}

LocalBundler::LocalBundler(int w, int h, const Eigen::Matrix3f &K) : K(K), width(w), height(h), frameCount(0)
{
    // This is for the reduced camera system
    FrameHessianR.create(96, 21 * NUM_KF * NUM_KF, CV_32FC1);
    FrameHessianRFinal.create(1, 21 * NUM_KF * NUM_KF, CV_32FC1);
    Residual.create(96, NUM_KF * 6, CV_32FC1);
    ResidualFinal.create(1, NUM_KF * 6, CV_32FC1);

    SafeCall(cudaMalloc((void **)&constPoseMat, sizeof(Sophus::SE3f) * NUM_KF * NUM_KF));
    SafeCall(cudaMalloc((void **)&frameData_dev, sizeof(Eigen::Vector3f) * w * h * NUM_KF));

    // initialize the frame array
    for (int i = 0; i < NUM_KF; ++i)
    {
        frame[i] = nullptr;
        frameHessian[i].create(96, 27, CV_32FC1);
    }

    Reset();
}

LocalBundler::~LocalBundler()
{
}

void LocalBundler::Reset()
{
}

__global__ void uploadFrameData_kernel(Eigen::Vector3f *frameData, int w, int h, cv::cuda::PtrStep<float> image)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    if (x == 0 || y == 0 || x == w - 1 || y == h - 1)
    {
        frameData[y * w + x] = Eigen::Vector3f(image.ptr(y)[x], 0, 0);
        return;
    }

    float gx = 0.5 * (image.ptr(y)[x + 1] - image.ptr(y)[x - 1]);
    float gy = 0.5 * (image.ptr(y + 1)[x] - image.ptr(y - 1)[x]);
    frameData[y * w + x] = Eigen::Vector3f(image.ptr(y)[x], gx, gy);
}

void LocalBundler::AddKeyFrame(unsigned long KFid, const cv::Mat depth, const cv::Mat image, const Sophus::SE3d &Tcw)
{
    FrameShell *F = new FrameShell();
    F->arrayIdx = -1;
    F->KFid = KFid;
    F->Tcw = Tcw;

    for (int i = 0; i < NUM_KF; ++i)
    {
        if (frame[i] == nullptr)
        {
            F->arrayIdx = i;
            frame[i] = F;
            framePose[i] = Tcw;
        }
    }

    if (F->arrayIdx == -1)
    {
        std::cout << "array idx non positive!" << std::endl;
    }

    frameCount++;
    F->image = cv::cuda::GpuMat(image);
    F->depth = cv::cuda::GpuMat(depth);

    // TODO: upload frame data to the global memory
    dim3 block(8, 8);
    dim3 grid(cv::divUp(width, block.x), cv::divUp(height, block.y));
    uploadFrameData_kernel<<<grid, block>>>(frameData_dev + F->arrayIdx * width * height, width, height, F->image);
    UpdatePoseMatrix(framePose);
}

__device__ Eigen::Vector3f GetInterpolateElement33(Eigen::Vector3f *I, float x, float y, int w)
{
    float dx = x - floor(x);
    float dy = y - floor(y);
    const int intx = static_cast<int>(x);
    const int inty = static_cast<int>(y);

    return (1 - dy) * (1 - dx) * I[inty * w + intx] +
           (1 - dy) * dx * I[inty * w + intx + 1] +
           dy * (1 - dx) * I[(inty + 1) * w + intx] +
           dy * dx * I[(inty + 1) * w + intx + 1];
}

__global__ void LinearizeAll_kernel(PointShell *points, int N,
                                    Eigen::Vector3f *frameData,
                                    cv::cuda::PtrStep<float> out,
                                    int w, int h,
                                    float fx, float fy,
                                    float cx, float cy)
{
    float sum[2] = {0, 0};

    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
        if (P.numResiduals == 0)
            continue;

        Eigen::Matrix<float, 2, 6> Jpdxi;
        Eigen::Matrix<float, 2, 3> Jhdp;

        for (int iRes = 0; iRes < NUM_KF; ++iRes)
        {
            RawResidual &res(P.res[iRes]);
            if (!res.active || iRes == P.hostIdx || res.state != OK)
                continue;

            const Sophus::SE3f &T(GetPoseFromMatrix(P.hostIdx, res.targetIdx));
            const float z = 1.0 / P.idepth;
            Eigen::Vector3f Point((P.x - cx) / fx, (P.y - cy) / fy, 1);
            Eigen::Vector3f PointScaled = T * Point * z;

            const float u = fx * PointScaled(0) / PointScaled(2) + cx;
            const float v = fy * PointScaled(1) / PointScaled(2) + cy;
            if (u < 1 || v < 1 || u >= w - 1 || v >= h - 1)
            {
                res.state = OOB;
                continue;
            }

            const float invz = 1.0 / PointScaled(2);
            const float invz2 = invz * invz;
            Jpdxi(0, 0) = fx * invz;
            Jpdxi(0, 1) = 0;
            Jpdxi(0, 2) = -fx * PointScaled(0) * PointScaled(1) * invz2;
            Jpdxi(0, 3) = fx + fx * PointScaled(0) * PointScaled(0) * invz2;
            Jpdxi(0, 4) = fx + fx * PointScaled(0) * PointScaled(0) * invz2;
            Jpdxi(0, 5) = -fx * PointScaled(0) * invz;
            Jpdxi(1, 0) = 0;
            Jpdxi(1, 1) = fy * invz;
            Jpdxi(1, 2) = -fy * PointScaled(1) * invz2;
            Jpdxi(1, 3) = -fy - fy * PointScaled(1) * PointScaled(1) * invz2;
            Jpdxi(1, 4) = fy * PointScaled(0) * PointScaled(1) * invz2;
            Jpdxi(1, 5) = fy * PointScaled(0) * invz;

            Eigen::Vector3f dI = GetInterpolateElement33(frameData + res.targetIdx * w * h, u, v, w);

            if (!isfinite(dI(0)) || !isfinite(dI(1)) || !isfinite(dI(2)))
                continue;

            Jhdp(0, 0) = fx * invz;
            Jhdp(0, 1) = 0;
            Jhdp(0, 2) = -PointScaled(0) * fx * invz2;
            Jhdp(1, 0) = 0;
            Jhdp(1, 1) = fy * invz;
            Jhdp(1, 2) = -PointScaled(1) * fy * invz2;

            Eigen::Vector3f PointRot = T.so3() * Point;

            res.JIdxi = dI.tail<2>().transpose() * Jpdxi;
            res.Jpdd = dI.tail<2>().transpose() * Jhdp * PointRot;
            res.r = dI(0) - P.intensity;
            res.hw = fabs(res.r) < 9 ? 1 : 9 / fabs(res.r);

            ++sum[1];
            sum[0] += res.hw * res.r * res.r;
        }
    }

    BlockReduceSum<float, 2>(sum);

    if (threadIdx.x == 0)
#pragma unroll
        for (int i = 0; i < 2; ++i)
            out.ptr(blockIdx.x)[i] = sum[i];
}

float LocalBundler::LineariseAll()
{
    dim3 block(1024);
    dim3 grid(cv::divUp(nPoints, block.x));
    LinearizeAll_kernel<<<grid, block>>>(points_dev, nPoints, frameData_dev, EnergySum,
                                         width, height, K(0, 0), K(1, 1), K(0, 2), K(1, 2));
    cv::cuda::reduce(EnergySum, EnergySumFinal, 0, cv::REDUCE_SUM);
    cv::Mat out(EnergySumFinal);
    return out.at<float>(0);
}

void LocalBundler::Marginalization()
{
}

__global__ void AccumulateFrameHessian_kernel(PointShell *points, int N,
                                              int targetIdx,
                                              cv::cuda::PtrStep<float> out)
{
    float sum[27];
    memset(&sum[0], 0, sizeof(float) * 27);

    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
        if (P.hostIdx == targetIdx || P.numResiduals == 0)
            continue;

        RawResidual &res(P.res[targetIdx]);
        if (!res.active)
            continue;

        int count = 0;
#pragma unroll
        for (int i = 0; i < 6; ++i)
        {
#pragma unroll
            for (int j = i; j < 7; ++j)
            {
                if (j == 6)
                    sum[count++] = res.hw * res.JIdxi(i) * res.r;
                else
                    sum[count++] = res.hw * res.JIdxi(i) * res.JIdxi(j);
            }
        }
    }

    BlockReduceSum<float, 27>(sum);

    if (threadIdx.x == 0)
#pragma unroll
        for (int k = 0; k < 27; ++k)
            out.ptr(blockIdx.x)[k] = sum[k];
}

void LocalBundler::AccumulateFrameHessian()
{
    cv::cuda::GpuMat hessianOut;
    int nKF = std::min(NUM_KF, frameCount);
    for (int i = 0; i < nKF; ++i)
    {
        AccumulateFrameHessian_kernel<<<96, 224>>>(points_dev, nPoints, i, frameHessian[i]);
        cv::cuda::reduce(frameHessian[i], hessianOut, 0, cv::REDUCE_SUM);
        cv::Mat hostData(hessianOut);
        RankUpdateHessian<6, 7>(hostData.ptr<float>(), frameHesOut[i].data(), frameResOut[i].data());
    }
}

__global__ void AccumulateShcurrHessian_kernel(PointShell *points, int N,
                                               int hostIdx, int targetIdx,
                                               cv::cuda::PtrStep<float> out)
{
    float sum[21];
    memset(&sum[0], 0, sizeof(float) * 21);

    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
        if (P.hostIdx == hostIdx || P.hostIdx == targetIdx || P.numResiduals == 0)
            continue;

        RawResidual &res(P.res[hostIdx]);
        RawResidual &res2(P.res[targetIdx]);
        if (!res.active || !res2.active)
            continue;

        float iHs = 1.0f / P.Hs;

        int count = 0;
#pragma unroll
        for (int i = 0; i < 6; ++i)
#pragma unroll
            for (int j = i; j < 6; ++j)
                sum[count++] = res.hw * res.JIdxi(i) * iHs * res2.hw * res2.JIdxi(j);
    }

    BlockReduceSum<float, 21>(sum);

    if (threadIdx.x == 0)
#pragma unroll
        for (int k = 0; k < 21; ++k)
            out.ptr(blockIdx.x)[k] = sum[k];
}

void LocalBundler::AccumulateShcurrHessian()
{
    int nKF = std::min(NUM_KF, frameCount);
    for (int i = 0; i < nKF; ++i)
        for (int j = i; j < nKF; ++j)
            AccumulateShcurrHessian_kernel<<<96, 224>>>(points_dev, nPoints, i, j, FrameHessianR);

    cv::cuda::reduce(FrameHessianR, FrameHessianRFinal, 0, cv::REDUCE_SUM);
}

__global__ void AccumulateShcurrResidual_kernel(PointShell *points, int N,
                                                int hostIdx,
                                                cv::cuda::PtrStep<float> out)
{
    float sum[6];
    memset(&sum[0], 0, sizeof(float) * 6);

    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
        if (P.hostIdx != hostIdx || P.numResiduals == 0)
            continue;

        float iHs = 1.0f / P.Hs;

#pragma unroll
        for (int i = 0; i < NUM_KF; ++i)
        {
            if (i == hostIdx)
                continue;

            RawResidual &res(P.res[i]);
            if (!res.active)
                continue;

#pragma unroll
            for (int i = 0; i < 6; ++i)
                sum[i] += res.hw * res.JIdxi(i) * res.Jpdd * iHs * res.r;
        }
    }

    BlockReduceSum<float, 6>(sum);

    if (threadIdx.x == 0)
#pragma unroll
        for (int k = 0; k < 6; ++k)
            out.ptr(blockIdx.x)[hostIdx * 6 + k] = sum[k];
}

void LocalBundler::AccumulateShcurrResidual()
{
    for (int i = 0; i < NUM_KF; ++i)
    {
        AccumulateShcurrResidual_kernel<<<96, 224>>>(points_dev, nPoints, i, Residual);
    }

    cv::cuda::reduce(Residual, ResidualFinal, 0, cv::REDUCE_SUM);
}

__global__ void AccumulatePointHessian_kernel(PointShell *points, int N)
{
    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
        if (P.numResiduals == 0)
            continue;

        for (int i = 0; i < NUM_KF; ++i)
        {
            RawResidual &res(P.res[i]);
            if (!res.active)
                continue;

            P.Hs += res.hw * res.Jpdd * res.Jpdd;
            P.bs += res.hw * res.Jpdd * res.r;
        }
    }
}

void LocalBundler::AccumulatePointHessian()
{
    // TODO: need to be called before accumulating frame hessian
    dim3 block(1024);
    dim3 grid(cv::divUp(nPoints, block.x));

    AccumulatePointHessian_kernel<<<grid, block>>>(points_dev, nPoints);
}

void LocalBundler::BundleAdjust(int maxIter)
{
    if (frameCount <= 2)
        return;
    else if (frameCount == 3)
        maxIter = 20;
    else if (frameCount == 4)
        maxIter = 15;

    float lastEnergy = LineariseAll();
    float bestEnergy = lastEnergy;

    for (int iter = 0; iter < maxIter; ++iter)
    {
        std::cout << "current energy: " << lastEnergy << std::endl;

        AccumulatePointHessian();
        AccumulateFrameHessian();
        AccumulateShcurrHessian();
        AccumulateShcurrResidual();

        SolveSystem();

        lastEnergy = LineariseAll();

        if (lastEnergy < bestEnergy)
        {
            bestEnergy = lastEnergy;
        }
    }
}

void LocalBundler::SolveSystem()
{
}

void LocalBundler::PopulateOccupancyGrid(FrameShell &F)
{
}
