#include "LocalBundler.h"
#include "CudaUtils.h"
#include "ImageProc.h"
#include "TrackingUtils.h"

__device__ Sophus::SE3f GetPoseFromMatrix(Sophus::SE3f *constPoseMat, int hostIdx, int targetIdx)
{
    return constPoseMat[hostIdx * NUM_KF + targetIdx];
}

__device__ void SetPoseMatrix(Sophus::SE3f *constPoseMat, const Sophus::SE3f &T, int hostIdx, int targetIdx)
{
    constPoseMat[hostIdx * NUM_KF + targetIdx] = T;
}

void LocalBundler::UpdatePoseMatrix(std::array<Sophus::SE3d, NUM_KF> &poses)
{
    Sophus::SE3f poses_host[NUM_KF * NUM_KF];
    for (int i = 0; i < poses.size(); ++i)
        for (int j = 0; j < poses.size(); ++j)
            if (i != j)
                poses_host[i * NUM_KF + j] = (poses[j].inverse() * poses[i]).cast<float>();

    SafeCall(cudaMemcpy(constPoseMat, &poses_host[0], sizeof(Sophus::SE3f) * NUM_KF * NUM_KF, cudaMemcpyHostToDevice));
}

__global__ void InitializePoints(PointShell *points, int *stack, int *ptr, int N)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= N)
        return;

    if (threadIdx.x == 0)
        ptr[0] = 0;

    stack[x] = x;

    PointShell &P(points[N]);
    P.hostIdx = -1;
    P.Hs = 0;
    P.bs = 0;
    P.numResiduals = 0;
#pragma unroll
    for (int i = 0; i < NUM_KF; ++i)
    {
        P.res[i].active = false;
        P.res[i].targetIdx = -1;
    }
}

LocalBundler::LocalBundler(int w, int h, const Eigen::Matrix3f &K)
    : K(K), width(w), height(h), frameCount(0), nPoints(0), lastKeyframeIdx(0)
{
    // This is for the reduced camera system
    FrameHessianR.create(96, 21 * NUM_KF * NUM_KF, CV_32FC1);
    FrameHessianRFinal.create(1, 21 * NUM_KF * NUM_KF, CV_32FC1);
    Residual.create(96, NUM_KF * 6, CV_32FC1);
    ResidualFinal.create(1, NUM_KF * 6, CV_32FC1);
    EnergySum.create(96, 2, CV_32FC1);
    EnergySumFinal.create(1, 2, CV_32FC1);

    SafeCall(cudaMalloc((void **)&N, sizeof(int)));
    SafeCall(cudaMemset(N, 0, sizeof(int)));

    SafeCall(cudaMalloc((void **)&stack, sizeof(int) * w * h));
    SafeCall(cudaMalloc((void **)&points_dev, sizeof(PointShell) * w * h));
    SafeCall(cudaMalloc((void **)&constPoseMat, sizeof(Sophus::SE3f) * NUM_KF * NUM_KF));
    SafeCall(cudaMalloc((void **)&frameData_dev, sizeof(Eigen::Vector3f) * w * h * NUM_KF));

    dim3 block(1024);
    dim3 grid(cv::divUp(w * h, block.x));
    InitializePoints<<<grid, block>>>(points_dev, stack, N, w * h);

    SafeCall(cudaDeviceSynchronize());
    SafeCall(cudaGetLastError());

    // initialize the frame array
    for (int i = 0; i < NUM_KF; ++i)
    {
        frame[i] = nullptr;
        frameHessian[i].create(96, 27, CV_32FC1);
        ReduceResidual[i].create(96, 6, CV_32FC1);
    }

    for (int i = 0; i < NUM_KF * NUM_KF; ++i)
    {
        reduceHessian[i].create(96, 21, CV_32FC1);
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
        // std::cout << (frame[i] == nullptr) << std::endl;

        if (!frame[i])
        {
            F->arrayIdx = i;
            frame[i] = F;
            framePose[i] = Tcw;
            lastKeyframeIdx = i;
            break;
        }
    }

    if (F->arrayIdx == -1)
    {
        std::cout << "array idx non positive!" << std::endl;
        return;
    }

    frameCount++;
    cv::Mat imFloat;
    image.convertTo(imFloat, CV_32FC1);

    F->image = cv::cuda::GpuMat(imFloat);
    F->depth = cv::cuda::GpuMat(depth);

    // TODO: upload frame data to the global memory
    dim3 block(8, 8);
    dim3 grid(cv::divUp(width, block.x), cv::divUp(height, block.y));
    uploadFrameData_kernel<<<grid, block>>>(frameData_dev + F->arrayIdx * width * height, width, height, F->image);

    SafeCall(cudaDeviceSynchronize());
    SafeCall(cudaGetLastError());

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
                                    Sophus::SE3f *poseMatrix,
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
        // printf("res: %d, frame: %d\n", P.numResiduals, P.hostIdx);

        Eigen::Matrix<float, 2, 6> Jpdxi;
        Eigen::Matrix<float, 2, 3> Jhdp;

        for (int iRes = 0; iRes < NUM_KF; ++iRes)
        {
            RawResidual &res(P.res[iRes]);
            if (!res.active || res.state != OK)
                continue;

            const Sophus::SE3f &T(GetPoseFromMatrix(poseMatrix, P.hostIdx, res.targetIdx));
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
            // printf("jpdd: %f\n", res.Jpdd);
            res.r = dI(0) - P.intensity;
            res.hw = fabs(res.r) < 9 ? 1 : 9 / fabs(res.r);

            // if (res.Jpdd > 100)
            //     printf("%f, %f,%f, %f,%f\n", PointRot(0), PointRot(1), PointRot(2), dI(1), dI(2));

            ++sum[1];
            // printf("id: %d, hw: %f, i: %f, i2: %f\n", x, res.hw, dI(0), P.intensity);
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
    if (nPoints == 0)
        return 0;

    // std::cout << "Now we have " << nPoints << " points" << std::endl;

    LinearizeAll_kernel<<<96, 224>>>(points_dev, nPoints, constPoseMat, frameData_dev, EnergySum,
                                     width, height, K(0, 0), K(1, 1), K(0, 2), K(1, 2));

    SafeCall(cudaDeviceSynchronize());
    SafeCall(cudaGetLastError());

    cv::cuda::reduce(EnergySum, EnergySumFinal, 0, cv::REDUCE_SUM);
    cv::Mat out(EnergySumFinal);
    // std::cout << "num residual: " << out.at<float>(1) << std::endl;
    return out.at<float>(0);
}

void LocalBundler::Marginalization()
{
}

__global__ void AccumulateFrameHessian_kernel(PointShell *points, int N,
                                              int frameIdx,
                                              cv::cuda::PtrStep<float> out)
{
    float sum[27];
    memset(&sum[0], 0, sizeof(float) * 27);

    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += gridDim.x * blockDim.x)
    {
        PointShell &P(points[x]);
        if (P.numResiduals == 0)
            continue;

        RawResidual &res(P.res[frameIdx]);
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
                    sum[count++] += res.hw * res.JIdxi(i) * res.r;
                else
                    sum[count++] += res.hw * res.JIdxi(i) * res.JIdxi(j);
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

        // std::cout << frameHesOut[i] << std::endl;

        SafeCall(cudaDeviceSynchronize());
        SafeCall(cudaGetLastError());
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
        if (P.numResiduals == 0)
            continue;

        RawResidual &res(P.res[hostIdx]);
        RawResidual &res2(P.res[targetIdx]);
        if (!res.active || !res2.active)
            continue;

        if (P.Hs == 0)
            continue;

        float iHs = 1.0 / P.Hs;

        // if (isnan(iHs) || !isfinite(iHs))
        //     printf("Hs: %a\n", iHs);

        int count = 0;
#pragma unroll
        for (int i = 0; i < 6; ++i)
#pragma unroll
            for (int j = i; j < 6; ++j)
                sum[count++] += res.hw * res.JIdxi(i) * iHs * res2.hw * res2.JIdxi(j);
    }

    BlockReduceSum<float, 21>(sum);

    if (threadIdx.x == 0)
#pragma unroll
        for (int k = 0; k < 21; ++k)
            out.ptr(blockIdx.x)[k] = sum[k];
}

void LocalBundler::AccumulateShcurrHessian()
{
    for (int i = 0; i < NUM_KF * NUM_KF; ++i)
        reduceHessian[i].setTo(0);

    cv::cuda::GpuMat out;
    int nKF = std::min(NUM_KF, frameCount);
    for (int i = 0; i < nKF; ++i)
    {
        for (int j = i; j < nKF; ++j)
        {
            AccumulateShcurrHessian_kernel<<<96, 224>>>(points_dev, nPoints, i, j, reduceHessian[i * NUM_KF + j]);

            SafeCall(cudaDeviceSynchronize());
            SafeCall(cudaGetLastError());

            cv::cuda::reduce(reduceHessian[i * NUM_KF + j], out, 0, cv::REDUCE_SUM);
            cv::Mat hostData(out);
            Eigen::Matrix<float, 6, 6> hessian;

            int count = 0;
            for (int ii = 0; ii < 6; ++ii)
                for (int jj = ii; jj < 6; ++jj)
                {
                    hessian(ii, jj) = hessian(jj, ii) = hostData.ptr<float>()[count++];
                }
            // std::cout << "i: " << i << " j: " << j << std::endl
            //           << hessian << std::endl;
            reduceHessianOut.block<6, 6>(i * 6, j * 6) = hessian;
            reduceHessianOut.block<6, 6>(j * 6, i * 6) = hessian.transpose();
        }
    }

    // std::cout << reduceHessianOut.topLeftCorner(nKF * 6, nKF * 6) << std::endl;

    // cv::cuda::reduce(FrameHessianR, FrameHessianRFinal, 0, cv::REDUCE_SUM);
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
        if (P.numResiduals == 0 || P.Hs == 0)
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
            out.ptr(blockIdx.x)[k] = sum[k];
}

void LocalBundler::AccumulateShcurrResidual()
{
    int nKF = std::min(NUM_KF, frameCount);
    for (int i = 0; i < nKF; ++i)
    {
        AccumulateShcurrResidual_kernel<<<96, 224>>>(points_dev, nPoints, i, ReduceResidual[i]);
        cv::cuda::reduce(ReduceResidual[i], ResidualFinal, 0, cv::REDUCE_SUM);

        cv::Mat out(ResidualFinal);
        // std::cout << out << std::endl;
        for (int k = 0; k < 6; ++k)
            ReduceResidualOut.middleRows<6>(i * 6)[k] = out.at<float>(k);

        SafeCall(cudaDeviceSynchronize());
        SafeCall(cudaGetLastError());
    }

    // std::cout << ReduceResidualOut.head<18>() << std::endl;

    // std::cout << ReduceResidualOut << std::endl;
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
            RawResidual &RES(P.res[i]);
            if (!RES.active)
                continue;

            P.Hs += RES.hw * RES.Jpdd * RES.Jpdd;
            P.bs += RES.hw * RES.Jpdd * RES.r;
        }
    }
}

void LocalBundler::AccumulatePointHessian()
{
    if (nPoints == 0)
        return;

    dim3 block(1024);
    dim3 grid(cv::divUp(nPoints, block.x));

    AccumulatePointHessian_kernel<<<grid, block>>>(points_dev, nPoints);

    SafeCall(cudaDeviceSynchronize());
    SafeCall(cudaGetLastError());
}

void LocalBundler::BundleAdjust(int maxIter)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    if (frameCount != 1)
    {
        FrameShell *F = frame[lastKeyframeIdx];
        if (F != nullptr)
            PopulateOccupancyGrid(F);
    }

    if (frameCount <= 2)
        return;
    else if (frameCount == 3)
        maxIter = 15;
    else if (frameCount == 4)
        maxIter = 10;

    float lastEnergy = LineariseAll();
    float energy = lastEnergy;
    float bestEnergy = lastEnergy;

    for (int iter = 0; iter < maxIter; ++iter)
    {
        std::cout << "current energy: " << energy << std::endl;

        AccumulatePointHessian();
        AccumulateFrameHessian();
        AccumulateShcurrHessian();
        AccumulateShcurrResidual();

        SolveSystem();

        float energy = LineariseAll();
        std::cout << "cost reduce: " << lastEnergy - energy << std::endl;

        if (lastEnergy < bestEnergy)
        {
            bestEnergy = lastEnergy;
        }

        lastEnergy = energy;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
}

__global__ void AllocatePoints_kernel(int *stack, int *N,
                                      PointShell *points_dev,
                                      int frameIdx, int w, int h,
                                      cv::cuda::PtrStep<char> grid,
                                      cv::cuda::PtrStep<float> image,
                                      cv::cuda::PtrStep<float> depth)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w - 1 || y >= h - 1)
        return;

    int u0 = x / 3;
    int v0 = y / 3;
    bool occupied = (grid.ptr(v0)[u0] == 1);

    if (occupied)
        return;

    float z = depth.ptr(y)[x];
    float im = image.ptr(y)[x];
    float dx = (image.ptr(y)[x + 1] - image.ptr(y)[x - 1]) * 0.5;
    float dy = (image.ptr(y + 1)[x] - image.ptr(y - 1)[x]) * 0.5;

    // printf("x: %d, y:%d, itensity: %f, dx: %f, dy: %f\n", x, y, im, dx, dy);

    if (z == z && im == im && z > 0.2 && z < 8.f && (dx * dx + dy * dy) > 4)
    {
        int idx = atomicAdd(N, 1);
        int realIdx = stack[idx];
        PointShell &P(points_dev[realIdx]);
        P.x = x;
        P.y = y;
        P.intensity = im;
        P.numResiduals = 0;
        P.idepth = 1.f / z;
        P.hostIdx = frameIdx;
    }
}

void LocalBundler::AllocatePoints()
{
    FrameShell *F = frame[lastKeyframeIdx];
    if (F == nullptr)
    {
        std::cerr << "FATAL: Frame is null" << std::endl;
        return;
    }

    if (F->KFid == 0)
    {
        F->OGrid.create(height / 3, width / 3, CV_8SC1);
        F->OGrid.setTo(0);
    }
    // else
    // {
    //     PopulateOccupancyGrid(F);
    // }

    // cv::Mat out(F->image);
    // double minval, maxval;
    // cv::minMaxIdx(out, &minval, &maxval);
    // std::cout << "min:" << minval << "max:" << maxval << std::endl;
    // cv::imshow("debug", out);
    // cv::waitKey(0);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(width, block.x), cv::divUp(height, block.y));
    AllocatePoints_kernel<<<grid, block>>>(stack, N, points_dev, F->arrayIdx, width, height, F->OGrid, F->image, F->depth);

    SafeCall(cudaDeviceSynchronize());
    SafeCall(cudaGetLastError());

    SafeCall(cudaMemcpy(&nPoints, N, sizeof(int), cudaMemcpyDeviceToHost));
}

Sophus::SE3d LocalBundler::GetLastKeyFramePose()
{
    FrameShell *F = frame[lastKeyframeIdx];
    return F->Tcw;
}

__global__ void BackSubstitution_kernel(PointShell *points, int N, float *x0)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= N)
        return;
    PointShell &P(points[x]);
    // printf("%d\n", P.numResiduals);

    if (P.numResiduals == 0 || P.Hs == 0)
        return;

    P.b1 = 0;
    for (int i = 1; i < NUM_KF; ++i)
    {
        RawResidual &RES(P.res[i]);

        if (!RES.active)
            continue;

        Eigen::Map<Eigen::Vector<float, 6>> x0i(&x0[(i - 1) * 6]);
        P.b1 += RES.hw * RES.JIdxi * RES.Jpdd * x0i;
    }

    P.x1 = P.b1 / P.Hs;
    // printf("%a\n", P.x1);
    P.idepth = 1.0 / (1.0 / P.idepth + P.x1);
}

void LocalBundler::SolveSystem()
{
    int nKF = std::min(NUM_KF, frameCount);
    Eigen::Matrix<float, -1, -1> H(nKF * 6, nKF * 6);
    H.setZero();
    for (int i = 0; i < nKF; ++i)
        H.block<6, 6>(i * 6, i * 6) = frameHesOut[i];
    H -= reduceHessianOut.topLeftCorner(nKF * 6, nKF * 6);

    Eigen::Vector<float, -1> b(nKF * 6);
    for (int i = 0; i < nKF; ++i)
        b.middleRows(i * 6, 6) = frameResOut[i];
    b -= ReduceResidualOut.leftCols(6 * nKF);

    auto realH = H.block(6, 6, (nKF - 1) * 6, (nKF - 1) * 6).cast<double>();
    auto realb = b.middleRows(6, (nKF - 1) * 6).cast<double>();

    Eigen::Vector<double, -1> x0 = realH.ldlt().solve(realb);
    Eigen::Vector<float, -1> x0_float = x0.cast<float>();

    float *x0_dev;
    SafeCall(cudaMalloc((void **)&x0_dev, sizeof(float) * (nKF - 1) * 6));
    SafeCall(cudaMemcpy(x0_dev, x0_float.data(), sizeof(float) * (nKF - 1) * 6, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid(cv::divUp(nPoints, block.x));
    BackSubstitution_kernel<<<grid, block>>>(points_dev, nPoints, x0_dev);

    SafeCall(cudaDeviceSynchronize());
    SafeCall(cudaGetLastError());

    for (int i = 1; i < nKF; ++i)
    {
        FrameShell *F = frame[i];
        if (F == nullptr)
            continue;

        Eigen::Vector<double, 6> x0i = x0.middleRows<6>((i - 1) * 6);
        F->Tcw = Sophus::SE3d::exp(x0i) * F->Tcw;
        framePose[i] = F->Tcw;
    }

    UpdatePoseMatrix(framePose);
}

__global__ void PopulateOccupancyGrid_kernel(PointShell *points, int N,
                                             Sophus::SE3f *poseMatrix,
                                             int frameIdx, int w, int h,
                                             float fx, float fy, float cx, float cy,
                                             cv::cuda::PtrStep<char> grid)
{
    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < N; x += blockDim.x * gridDim.x)
    {
        PointShell &P(points[x]);
        Sophus::SE3f T(GetPoseFromMatrix(poseMatrix, P.hostIdx, frameIdx));

        float z = 1.0 / P.idepth;
        Eigen::Vector3f pt = T * Eigen::Vector3f(z * (P.x - cx) / fx, z * (P.y - cy) / fy, z);
        float u = fx * pt(0) / pt(2) + cx;
        float v = fy * pt(1) / pt(2) + cy;
        // printf("u: %f, v: %f\n", u, v);

        if (u < 2 || v < 2 || u > w - 2 || v > h - 2)
            continue;
        P.numResiduals++;

        RawResidual &RES(P.res[frameIdx]);
        RES.active = true;
        RES.state = OK;
        RES.targetIdx = frameIdx;

        int u0 = static_cast<int>(u) / 3;
        int v0 = static_cast<int>(v) / 3;
        grid.ptr(v0)[u0] = 1;
    }
}

void LocalBundler::PopulateOccupancyGrid(FrameShell *F)
{
    F->OGrid.create(height / 3, width / 3, CV_8SC1);
    F->OGrid.setTo(0);

    PopulateOccupancyGrid_kernel<<<96, 224>>>(points_dev, nPoints, constPoseMat, F->arrayIdx,
                                              width, height, K(0, 0), K(1, 1),
                                              K(0, 2), K(1, 2), F->OGrid);

    SafeCall(cudaDeviceSynchronize());
    SafeCall(cudaGetLastError());
}
