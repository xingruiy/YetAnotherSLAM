#include "CoarseTracking.h"
#include "TrackingUtils.h"
#include "ImageProc.h"

CoarseTracking::CoarseTracking(int w, int h, Eigen::Matrix3f K, bool bRGB, bool bIcp)
{
    if (bRGB && bIcp)
        mModal = TrackingModal::RGB_AND_DEPTH;
    else if (bRGB)
        mModal = TrackingModal::RGB_ONLY;
    else
        mModal = TrackingModal::DEPTH_ONLY;

    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        int wLvl = w >> lvl;
        int hLvl = h >> lvl;

        mK[lvl] = K / (1 << lvl);
        mK[lvl](2, 2) = 1.0f;

        mvWidth[lvl] = wLvl;
        mvHeight[lvl] = hLvl;
    }

    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        int wLvl = mvWidth[lvl];
        int hLvl = mvHeight[lvl];

        currDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        refDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        mvCurrentIntensity[lvl].create(hLvl, wLvl, CV_32FC1);
        mvReferenceIntensity[lvl].create(hLvl, wLvl, CV_32FC1);
        mvIntensityGradientX[lvl].create(hLvl, wLvl, CV_32FC1);
        mvIntensityGradientY[lvl].create(hLvl, wLvl, CV_32FC1);
        mvReferencePointTransformed[lvl].create(hLvl, wLvl, CV_32FC4);

        mvCurrentInvDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        mvReferenceInvDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        mvInvDepthGradientX[lvl].create(hLvl, wLvl, CV_32FC1);
        mvInvDepthGradientY[lvl].create(hLvl, wLvl, CV_32FC1);

        mvCurrentVMap[lvl].create(hLvl, wLvl, CV_32FC4);
        mvCurrentNMap[lvl].create(hLvl, wLvl, CV_32FC4);
        mvReferenceVMap[lvl].create(hLvl, wLvl, CV_32FC4);
        mvReferenceNMap[lvl].create(hLvl, wLvl, CV_32FC4);

        mvCurrentCurvature[lvl].create(hLvl, wLvl, CV_32FC1);
        mvReferenceCurvature[lvl].create(hLvl, wLvl, CV_32FC1);
    }

    // Create temporary buffers
    mGpuBufferFloat96x29.create(96, 29, CV_32FC1);
    mGpuBufferFloat96x3.create(96, 3, CV_32FC1);
    mGpuBufferFloat96x2.create(96, 2, CV_32FC1);
    mGpuBufferFloat96x1.create(96, 1, CV_32FC1);
    mGpuBufferFloat1x29.create(1, 29, CV_32FC1);
    mGpuBufferFloat1x3.create(1, 2, CV_32FC1);
    mGpuBufferFloat1x2.create(1, 2, CV_32FC1);
    mGpuBufferFloat1x1.create(1, 1, CV_32FC1);
    mGpuBufferVector4HxW.create(h, w, CV_32FC4);
    mGpuBufferVector7HxW.create(h, w, CV_32FC(7));
    mGpuBufferRawDepth.create(h, w, CV_32FC1);
}

// __global__ void SelectPoints_kernel(cv::cuda::PtrStepSz<float> im,
//                                     cv::cuda::PtrStep<float> depth,
//                                     FramePoint *pts, int *N)
// {
//     int x = blockDim.x * blockIdx.x + threadIdx.x;
//     int y = blockDim.y * blockIdx.y + threadIdx.y;
//     if (x < 3 || y < 3 || x >= im.cols - 3 || y >= im.rows - 3)
//         return;

//     float sample = im.ptr(y)[x];
//     float idepth = 1.0f / depth.ptr(y)[x];

//     if (!isnan(idepth) && isfinite(idepth) && isfinite(sample))
//     {
//     }
// }

void CoarseTracking::SetReferenceImage(const cv::Mat &imGray)
{
    cv::Mat imGrayFloat;
    imGray.convertTo(imGrayFloat, CV_32FC1);

    // auto t1 = std::chrono::high_resolution_clock::now();
    // int w = imGray.cols;
    // int h = imGray.rows;
    // float *pImage = (float *)aligned_alloc(16, w * h * 4 * sizeof(float));
    // __m128 *pImageSSE = (__m128 *)pImage;
    // for (int i = 0; i < w * h; ++i)
    // {
    //     pImageSSE[i] = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
    // }

    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::cout << "time cost: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        if (lvl == 0)
            mvReferenceIntensity[0].upload(imGrayFloat);
        else
            // cv::cuda::pyrDown(mvReferenceIntensity[lvl - 1], mvReferenceIntensity[lvl]);
            PyrDownImage(mvReferenceIntensity[lvl - 1], mvReferenceIntensity[lvl]);
    }
}

__global__ void MakeGradientKernel(int w, int h,
                                   cv::cuda::PtrStep<float> img,
                                   cv::cuda::PtrStep<float> grad2)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x == 0 || y == 0 || x >= w - 1 || y >= h - 1)
        return;

    float dx = (img.ptr(y)[x + 1] - img.ptr(y)[x - 1]) * 0.5f;
    float dy = (img.ptr(y + 1)[x] - img.ptr(y - 1)[x]) * 0.5f;
    grad2.ptr(y)[x] = dx * dx + dy * dy;
}

void CoarseTracking::setKeyFrame(cv::Mat img, cv::Mat depth)
{
    int w = img.cols;
    int h = img.rows;
    keyframeImage.upload(img);
    keyframeDepth.upload(depth);
    if (keyframeAbsGrab2.empty())
        keyframeAbsGrab2.create(h, w, CV_32FC1);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(w, block.x), cv::divUp(h, block.y));
    MakeGradientKernel<<<grid, block>>>(w, h, keyframeImage, keyframeAbsGrab2);
}

__global__ void computeOpticalFlow(int w, int h, float fx, float fy,
                                   float ifx, float ify, float cx, float cy,
                                   Eigen::Matrix3f R, Eigen::Vector3f t,
                                   cv::cuda::PtrStep<float> flowVec,
                                   cv::cuda::PtrStep<float> grad,
                                   cv::cuda::PtrStep<float> depth)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    if (x == 0 || y == 0 || x == w || y == h)
    {
        flowVec.ptr(y * w + x)[0] = 0;
        flowVec.ptr(y * w + x)[1] = 0;
        return;
    }

    if (grad.ptr(y)[x] <= 1)
    {
        flowVec.ptr(y * w + x)[0] = 0;
        flowVec.ptr(y * w + x)[1] = 0;
        return;
    }

    float z = depth.ptr(y)[x];
    if (z >= 0 && z < 8.f && !isnan(z))
    {
        Eigen::Vector3f pt(ifx * (x - cx) * z, ifx * (y - cy) * z, z);
        Eigen::Vector3f ptRt = R * pt + t;
        Eigen::Vector3f ptt = pt + t;

        float u = fx * ptRt[0] / ptRt[2] + cx;
        float v = fy * ptRt[1] / ptRt[2] + cy;
        float u2 = fx * ptt[0] / ptt[2] + cx;
        float v2 = fy * ptt[1] / ptt[2] + cy;

        float flow = (x - u) * (x - u) + (y - v) * (y - v);
        float flow2 = (x - u2) * (x - u2) + (y - v2) * (y - v2);

        flowVec.ptr(y * w + x)[0] = flow;
        flowVec.ptr(y * w + x)[1] = flow2;
    }
    else
    {
        flowVec.ptr(y * w + x)[0] = 0;
        flowVec.ptr(y * w + x)[1] = 0;
    }
}

bool CoarseTracking::needNewKF(const Sophus::SE3d &kf2F)
{
    float fx = mK[0](0, 0);
    float fy = mK[0](1, 1);
    float cx = mK[0](0, 2);
    float cy = mK[0](1, 2);
    float ifx = 1.0f / fx;
    float ify = 1.0f / fy;
    int w = mvWidth[0];
    int h = mvHeight[0];
    Eigen::Matrix3f R = kf2F.rotationMatrix().cast<float>();
    Eigen::Vector3f t = kf2F.translation().cast<float>();
    cv::cuda::GpuMat flow(w * h, 2, CV_32FC1);
    flow.setTo(0);
    dim3 block(8, 8);
    dim3 grid(cv::divUp(w, block.x), cv::divUp(h, block.y));

    computeOpticalFlow<<<grid, block>>>(w, h, fx, fy, ifx, ify, cx, cy, R, t, flow, keyframeAbsGrab2, keyframeDepth);
    cv::cuda::GpuMat flowSum;
    cv::cuda::reduce(flow, flowSum, 0, cv::REDUCE_SUM);
    cv::Mat hostData(flowSum);
    Eigen::Vector2f flowVec;

    flowVec[0] = hostData.ptr<float>(0)[0];
    flowVec[1] = hostData.ptr<float>(0)[1];

    // std::cout << flowVec.transpose() << std::endl;
    float weight_Rt = 0.02; //* (w + h);
    float weight_t = 0.04;  // * (w + h);
    float rt = weight_Rt * sqrtf(flowVec[0]) / (w * h) + weight_t * sqrtf(flowVec[1]) / (w * h);
    std::cout << rt << std::endl;
    return (weight_Rt * flowVec[0] / (w + h) + weight_t * flowVec[1] / (w + h)) > 1;
}

void CoarseTracking::SetReferenceDepth(const cv::Mat &imDepth)
{
    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        if (lvl == 0)
        {
            mGpuBufferRawDepth.upload(imDepth);
            DepthToInvDepth(mGpuBufferRawDepth, mvReferenceInvDepth[lvl]);
        }
        else
            PyrDownDepth(mvReferenceInvDepth[lvl - 1], mvReferenceInvDepth[lvl]);

        float invfx = 1.0 / mK[lvl](0, 0);
        float invfy = 1.0 / mK[lvl](1, 1);
        float cx = mK[lvl](0, 2);
        float cy = mK[lvl](1, 2);

        ComputeVertexMap(mvReferenceInvDepth[lvl], mvReferenceVMap[lvl], invfx, invfy, cx, cy, 3.0f);
        ComputeNormalMap(mvReferenceVMap[lvl], mvReferenceNMap[lvl]);
        ComputeCurvature(mvReferenceVMap[lvl], mvReferenceNMap[lvl], mvReferenceCurvature[lvl]);
        // ComputeNormalAndMeanCurvature(mvReferenceVMap[lvl], mvReferenceNMap[lvl], mvReferenceCurvature[lvl]);
    }

    // cv::cuda::GpuMat curvature(mvReferenceVMap[0].size(), CV_32FC1);
    // cv::Mat out(mvReferenceNMap[0]);
    // cv::imshow("out", out);
    // cv::waitKey(0);
}

void CoarseTracking::SetTrackingImage(const cv::Mat &imGray)
{
    cv::Mat imGrayFloat;
    imGray.convertTo(imGrayFloat, CV_32FC1);

    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        if (lvl == 0)
            mvCurrentIntensity[lvl].upload(imGrayFloat);
        else
            // cv::cuda::pyrDown(mvCurrentIntensity[lvl - 1], mvCurrentIntensity[lvl]);
            PyrDownImage(mvCurrentIntensity[lvl - 1], mvCurrentIntensity[lvl]);

        ComputeImageGradientCentralDifference(mvCurrentIntensity[lvl], mvIntensityGradientX[lvl], mvIntensityGradientY[lvl]);
    }
}

void CoarseTracking::SetTrackingDepth(const cv::Mat &imDepth)
{
    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        if (lvl == 0)
        {
            mGpuBufferRawDepth.upload(imDepth);
            DepthToInvDepth(mGpuBufferRawDepth, mvCurrentInvDepth[lvl]);
        }
        else
            PyrDownDepth(mvCurrentInvDepth[lvl - 1], mvCurrentInvDepth[lvl]);

        float invfx = 1.0 / mK[lvl](0, 0);
        float invfy = 1.0 / mK[lvl](1, 1);
        float cx = mK[lvl](0, 2);
        float cy = mK[lvl](1, 2);

        ComputeVertexMap(mvCurrentInvDepth[lvl], mvCurrentVMap[lvl], invfx, invfy, cx, cy, 3.0f);
        ComputeNormalMap(mvCurrentVMap[lvl], mvCurrentNMap[lvl]);
        ComputeCurvature(mvCurrentVMap[lvl], mvCurrentNMap[lvl], mvCurrentCurvature[lvl]);

        // ComputeNormalAndMeanCurvature(mvCurrentVMap[lvl], mvCurrentNMap[lvl], mvCurrentCurvature[lvl]);
    }
}

void CoarseTracking::SetReferenceModel(const cv::cuda::GpuMat vmap)
{
    vmap.copyTo(mvReferenceVMap[0]);
    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        if (lvl != 0)
        { // cv::cuda::pyrDown(mvReferenceVMap[lvl - 1], mvReferenceVMap[lvl]);
            PyrDownVec4f(mvReferenceVMap[lvl - 1], mvReferenceVMap[lvl]);
            // cv::Mat out(mvReferenceVMap[lvl]);
            // cv::imshow("out", out);
            // cv::waitKey(0);
        }

        ComputeNormalMap(mvReferenceVMap[lvl], mvReferenceNMap[lvl]);
        ComputeCurvature(mvReferenceVMap[lvl], mvReferenceNMap[lvl], mvReferenceCurvature[lvl]);
    }
}

Sophus::SE3d CoarseTracking::GetTransform(const Sophus::SE3d &init, const bool bSwapBuffer)
{
    int nIteration = 0;
    int nSuccessfulIteration = 0;

    Sophus::SE3d estimate = init;
    Sophus::SE3d lastSuccessEstimate = estimate;
    std::vector<int> vIterations = {10, 10, 15, 15, 15, 15};

    for (int lvl = NUM_PYR - 1; lvl >= 0; --lvl)
    {
        float lastError = std::numeric_limits<float>::max();
        for (int iter = 0; iter < vIterations[lvl]; ++iter)
        {
            Eigen::Matrix<float, 6, 6> hessian = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Matrix<float, 6, 1> residual = Eigen::Matrix<float, 6, 1>::Zero();

            switch (mModal)
            {
            case TrackingModal::RGB_ONLY:
                ComputeSingleStepRGB(lvl, estimate, hessian.data(), residual.data());
                break;

            case TrackingModal::DEPTH_ONLY:
                ComputeSingleStepDepth(lvl, estimate, hessian.data(), residual.data());
                break;

            case TrackingModal::RGB_AND_DEPTH:
                ComputeSingleStepRGBDLinear(lvl, estimate, hessian.data(), residual.data());
                break;
            }

            float error = sqrt(residualSum) / (numResidual + 6);
            Eigen::Matrix<double, 6, 1> update = hessian.cast<double>().ldlt().solve(residual.cast<double>());

            if (std::isnan(update(0)))
            {
                mbTrackingGood = false;
                return Sophus::SE3d();
            }

            // update = ClampEigenVector(update, 0.05, -0.05);

            if (update.norm() < 1e-3)
                break;

            estimate = Sophus::SE3d::exp(update) * estimate;
            if (error < lastError)
            {
                lastSuccessEstimate = estimate;
                lastError = error;
                nSuccessfulIteration++;
            }

            nIteration++;
        }
    }

    if (bSwapBuffer)
    {
        SwapFrameBuffer();
    }

    mbTrackingGood = true;
    return lastSuccessEstimate;
}

void CoarseTracking::TransformReferencePoint(const int lvl, const Sophus::SE3d &T)
{
    auto refInvDepth = mvReferenceInvDepth[lvl];
    auto refPtTransformedLvl = mvReferencePointTransformed[lvl];
    auto KLvl = mK[lvl];

    ::TransformReferencePoint(refInvDepth, refPtTransformedLvl, KLvl, T);
}

void CoarseTracking::ComputeSingleStepRGB(
    const int lvl,
    const Sophus::SE3d &T,
    float *hessian,
    float *residual)
{
    TransformReferencePoint(lvl, T);

    const int w = mvWidth[lvl];
    const int h = mvHeight[lvl];

    se3StepRGBResidualFunctor functor;
    functor.w = w;
    functor.h = h;
    functor.n = w * h;
    functor.refInt = mvReferenceIntensity[lvl];
    functor.currInt = mvCurrentIntensity[lvl];
    functor.currGx = mvIntensityGradientX[lvl];
    functor.currGy = mvIntensityGradientY[lvl];
    functor.refPtWarped = mvReferencePointTransformed[lvl];
    functor.refResidual = mGpuBufferVector4HxW;
    functor.fx = mK[lvl](0, 0);
    functor.fy = mK[lvl](1, 1);
    functor.cx = mK[lvl](0, 2);
    functor.cy = mK[lvl](1, 2);
    functor.out = mGpuBufferFloat96x2;

    callDeviceFunctor<<<96, 224>>>(functor);
    cv::cuda::reduce(mGpuBufferFloat96x2, mGpuBufferFloat1x2, 0, cv::REDUCE_SUM);
    cv::Mat hostData(mGpuBufferFloat1x2);

    iResidualSum = hostData.ptr<float>(0)[0];
    numResidual = hostData.ptr<float>(0)[1];

    VarianceEstimator estimator;
    estimator.w = w;
    estimator.h = h;
    estimator.n = w * h;
    estimator.meanEstimated = iResidualSum / numResidual;
    estimator.residual = mGpuBufferVector4HxW;
    estimator.out = mGpuBufferFloat96x1;

    callDeviceFunctor<<<96, 224>>>(estimator);
    cv::cuda::reduce(mGpuBufferFloat96x1, mGpuBufferFloat1x1, 0, cv::REDUCE_SUM);
    mGpuBufferFloat1x1.download(hostData);

    float squaredDeviationSum = hostData.ptr<float>(0)[0];
    float varEstimated = sqrt(squaredDeviationSum / (numResidual - 1));

    se3StepRGBFunctor sfunctor;
    sfunctor.w = w;
    sfunctor.h = h;
    sfunctor.n = w * h;
    sfunctor.huberTh = 4.685 * varEstimated;
    sfunctor.refPtWarped = mvReferencePointTransformed[lvl];
    sfunctor.refResidual = mGpuBufferVector4HxW;
    sfunctor.fx = mK[lvl](0, 0);
    sfunctor.fy = mK[lvl](1, 1);
    sfunctor.out = mGpuBufferFloat96x29;

    callDeviceFunctor<<<96, 224>>>(sfunctor);
    cv::cuda::reduce(mGpuBufferFloat96x29, mGpuBufferFloat1x29, 0, cv::REDUCE_SUM);

    mGpuBufferFloat1x29.download(hostData);
    RankUpdateHessian<6, 7>(hostData.ptr<float>(0), hessian, residual);

    residualSum = hostData.ptr<float>(0)[27];
}

void CoarseTracking::SwapFrameBuffer()
{
    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        std::swap(mvReferenceVMap[lvl], mvCurrentVMap[lvl]);
        std::swap(mvReferenceNMap[lvl], mvCurrentNMap[lvl]);
        std::swap(mvReferenceInvDepth[lvl], mvCurrentInvDepth[lvl]);
        std::swap(mvReferenceIntensity[lvl], mvCurrentIntensity[lvl]);
    }
}

struct IcpStepFunctor
{
    cv::cuda::PtrStep<Eigen::Vector4f> vmap_curr;
    cv::cuda::PtrStep<Eigen::Vector4f> nmap_curr;
    cv::cuda::PtrStep<Eigen::Vector4f> vmap_last;
    cv::cuda::PtrStep<Eigen::Vector4f> nmap_last;
    cv::cuda::PtrStep<float> curv_last;
    cv::cuda::PtrStep<float> curv_curr;
    int cols, rows, N;
    float fx, fy, cx, cy;
    float angleTH, distTH;
    Sophus::SE3f T_last_curr;
    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ bool ProjectPoint(int &x, int &y,
                                                 Eigen::Vector3f &v_curr,
                                                 Eigen::Vector3f &n_last,
                                                 Eigen::Vector3f &v_last) const;
    __device__ __forceinline__ void GetProduct(int &k, float *out) const;
    __device__ __forceinline__ void operator()() const;
};

__device__ __forceinline__ bool IcpStepFunctor::ProjectPoint(int &x, int &y,
                                                             Eigen::Vector3f &v_curr,
                                                             Eigen::Vector3f &n_last,
                                                             Eigen::Vector3f &v_last) const
{
    Eigen::Vector4f v_last_c = vmap_last.ptr(y)[x];
    if (v_last_c(3) < 0)
        return false;

    v_last = T_last_curr * v_last_c.head<3>();

    float invz = 1.0 / v_last(2);
    int u = __float2int_rd(fx * v_last(0) * invz + cx + 0.5);
    int v = __float2int_rd(fy * v_last(1) * invz + cy + 0.5);
    if (u < 1 || v < 1 || u >= cols - 1 || v >= rows - 1)
        return false;

    Eigen::Vector4f v_curr_c = vmap_curr.ptr(v)[u];
    v_curr = v_curr_c.head<3>();
    if (v_curr_c(3) < 0)
        return false;

    Eigen::Vector4f n_last_c = nmap_last.ptr(y)[x];
    n_last = T_last_curr.so3() * n_last_c.head<3>();

    Eigen::Vector4f n_curr_c = nmap_curr.ptr(v)[u];

    float dist = (v_last - v_curr).norm();
    float angle = n_curr_c.head<3>().dot(n_last);

    float c_last = curv_last.ptr(y)[x];
    float c_curr = curv_curr.ptr(v)[u];
    float cdiff = fabs(log(c_last) - log(c_curr));

    return (angle >= 0.6 && dist < 0.3 && cdiff < 2 && n_last_c(3) > 0 && n_curr_c(3) > 0);
    // return (angle < angleTH && dist < distTH && cdiff < 2 && n_last_c(3) > 0 && n_curr_c(3) > 0);
}

__device__ __forceinline__ void IcpStepFunctor::GetProduct(int &k, float *sum) const
{
    int y = k / cols;
    int x = k - (y * cols);

    Eigen::Vector3f v_curr, n_last, v_last;
    float row[7] = {0, 0, 0, 0, 0, 0, 0};
    bool found = ProjectPoint(x, y, v_curr, n_last, v_last);

    if (found)
    {
        row[6] = n_last.dot(v_curr - v_last);
        float hw = 1; //fabs(row[6]) < 0.3 ? 1 : 0.3 / fabs(row[6]);
        row[6] *= hw;
        *(Eigen::Vector3f *)&row[0] = hw * n_last;
        *(Eigen::Vector3f *)&row[3] = hw * v_last.cross(n_last);
    }

    int count = 0;
#pragma unroll
    for (int i = 0; i < 7; ++i)
#pragma unroll
        for (int j = i; j < 7; ++j)
            sum[count++] = row[i] * row[j];
    sum[count] = (float)found;
}

__device__ __forceinline__ void IcpStepFunctor::operator()() const
{
    float sum[29] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float val[29];
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += blockDim.x * gridDim.x)
    {
        GetProduct(k, val);

#pragma unroll
        for (int i = 0; i < 29; ++i)
        {
            sum[i] += val[i];
        }
    }

    BlockReduceSum<float, 29>(sum);

    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int i = 0; i < 29; ++i)
            out.ptr(blockIdx.x)[i] = sum[i];
    }
}

void CoarseTracking::ComputeSingleStepDepth(
    const int lvl,
    const Sophus::SE3d &T,
    float *hessian,
    float *residual)
{
    int cols = mvWidth[lvl];
    int rows = mvHeight[lvl];

    IcpStepFunctor icpStep;
    icpStep.out = mGpuBufferFloat96x29;
    icpStep.vmap_curr = mvCurrentVMap[lvl];
    icpStep.nmap_curr = mvCurrentNMap[lvl];
    icpStep.vmap_last = mvReferenceVMap[lvl];
    icpStep.nmap_last = mvReferenceNMap[lvl];
    icpStep.curv_last = mvReferenceCurvature[lvl];
    icpStep.curv_curr = mvCurrentCurvature[lvl];
    icpStep.cols = cols;
    icpStep.rows = rows;
    icpStep.N = cols * rows;
    icpStep.T_last_curr = T.cast<float>();
    icpStep.angleTH = sin(20.f * 3.14159254f / 180.f);
    icpStep.distTH = 0.01;
    icpStep.fx = mK[lvl](0, 0);
    icpStep.fy = mK[lvl](1, 1);
    icpStep.cx = mK[lvl](0, 2);
    icpStep.cy = mK[lvl](1, 2);

    callDeviceFunctor<<<96, 224>>>(icpStep);
    cv::cuda::reduce(mGpuBufferFloat96x29, mGpuBufferFloat1x29, 0, cv::REDUCE_SUM);

    cv::Mat hostData(mGpuBufferFloat1x29);
    RankUpdateHessian<6, 7>(hostData.ptr<float>(0), hessian, residual);

    residualSum = hostData.ptr<float>(0)[27];
}

void CoarseTracking::ComputeSingleStepRGBD(
    const int lvl,
    const Sophus::SE3d &T,
    float *hessian,
    float *residual)
{
    TransformReferencePoint(lvl, T);

    const int w = mvWidth[lvl];
    const int h = mvHeight[lvl];

    se3StepRGBDResidualFunctor functor;
    functor.w = w;
    functor.h = h;
    functor.n = w * h;
    functor.refInt = mvReferenceIntensity[lvl];
    functor.currInt = mvCurrentIntensity[lvl];
    functor.currGx = mvIntensityGradientX[lvl];
    functor.currGy = mvIntensityGradientY[lvl];
    functor.currInvDepth = mvCurrentInvDepth[lvl];
    functor.currInvDepthGx = mvInvDepthGradientX[lvl];
    functor.currInvDepthGy = mvInvDepthGradientY[lvl];
    functor.refPtWarped = mvReferencePointTransformed[lvl];
    functor.refResidual = mGpuBufferVector7HxW;
    functor.fx = mK[lvl](0, 0);
    functor.fy = mK[lvl](1, 1);
    functor.cx = mK[lvl](0, 2);
    functor.cy = mK[lvl](1, 2);
    functor.out = mGpuBufferFloat96x3;

    callDeviceFunctor<<<96, 224>>>(functor);
    cv::cuda::reduce(mGpuBufferFloat96x3, mGpuBufferFloat1x3, 0, cv::REDUCE_SUM);
    cv::Mat hostData(mGpuBufferFloat1x3);

    float iResidualSum = hostData.ptr<float>(0)[0];
    float dResidualSum = hostData.ptr<float>(0)[1];
    numResidual = hostData.ptr<float>(0)[2];

    VarCov2DEstimator estimator;
    estimator.h = h;
    estimator.w = w;
    estimator.n = h * w;
    estimator.meanEstimated = Eigen::Vector2f(iResidualSum, dResidualSum) / numResidual;
    estimator.residual = mGpuBufferVector7HxW;
    estimator.out = mGpuBufferFloat96x3;

    callDeviceFunctor<<<96, 224>>>(estimator);
    cv::cuda::reduce(mGpuBufferFloat96x3, mGpuBufferFloat1x3, 0, cv::REDUCE_SUM);
    mGpuBufferFloat1x3.download(hostData);

    Eigen::Matrix2f varEstimated;
    varEstimated(0, 0) = hostData.ptr<float>(0)[0];
    varEstimated(1, 1) = hostData.ptr<float>(0)[1];
    varEstimated(0, 1) = varEstimated(1, 0) = hostData.ptr<float>(0)[2];
    varEstimated /= (numResidual - 1);

    se3StepRGBDFunctor sfunctor;
    sfunctor.w = w;
    sfunctor.h = h;
    sfunctor.n = w * h;
    sfunctor.stddevI = 1.345 * varEstimated(0, 0);
    sfunctor.stddevD = 4.685 * varEstimated(1, 1);
    sfunctor.precision = varEstimated.inverse();
    sfunctor.refPtWarped = mvReferencePointTransformed[lvl];
    sfunctor.refResidual = mGpuBufferVector7HxW;
    sfunctor.fx = mK[lvl](0, 0);
    sfunctor.fy = mK[lvl](1, 1);
    sfunctor.out = mGpuBufferFloat96x29;

    callDeviceFunctor<<<96, 224>>>(sfunctor);
    cv::cuda::reduce(mGpuBufferFloat96x29, mGpuBufferFloat1x29, 0, cv::REDUCE_SUM);

    mGpuBufferFloat1x29.download(hostData);
    RankUpdateHessian<6, 7>(hostData.ptr<float>(0), hessian, residual);

    residualSum = hostData.ptr<float>(0)[27];
}

void CoarseTracking::ComputeSingleStepRGBDLinear(
    const int lvl,
    const Sophus::SE3d &T,
    float *hessian,
    float *residual)
{
    Eigen::Map<Eigen::Matrix<float, 6, 6>> hessianMapped(hessian);
    Eigen::Map<Eigen::Matrix<float, 6, 1>> residualMapped(residual);

    Eigen::Matrix<float, 6, 6> hessianBuffer;
    Eigen::Matrix<float, 6, 1> residualBuffer;

    ComputeSingleStepRGB(lvl, T, hessianBuffer.data(), residualBuffer.data());

    hessianMapped += hessianBuffer;
    residualMapped += residualBuffer;

    hessianBuffer.setZero();
    residualBuffer.setZero();

    ComputeSingleStepDepth(lvl, T, hessianBuffer.data(), residualBuffer.data());

    hessianMapped += 100 * hessianBuffer;
    residualMapped += 10 * residualBuffer;

    mHessian = hessianMapped;
}

cv::cuda::GpuMat CoarseTracking::GetReferenceDepth(const int lvl) const
{
    return mGpuBufferRawDepth;
}

void CoarseTracking::WriteDebugImages()
{
    cv::Mat out;
    mvCurrentIntensity[0].download(out);
    cv::imwrite("curr_image.png", out);
    mvReferenceIntensity[0].download(out);
    cv::imwrite("last_image.png", out);
    mvIntensityGradientX[0].download(out);
    cv::imwrite("gx.png", out);
    mvIntensityGradientY[0].download(out);
    cv::imwrite("gy.png", out);
    cv::imshow("out", out);
    cv::waitKey(0);
}

Eigen::Matrix<double, 6, 6> CoarseTracking::GetCovarianceMatrix()
{
    return mHessian.cast<double>().lu().inverse();
}