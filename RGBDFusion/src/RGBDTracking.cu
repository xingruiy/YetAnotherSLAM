#include "RGBDTracking.h"
#include "TrackingUtils.h"
#include "ImageProc.h"

RGBDTracking::RGBDTracking(const int &w, const int &h,
                           const Eigen::Matrix3d &K,
                           const bool &bRGB, const bool &bDepth)
{
    if (bRGB && bDepth)
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

        mvCurrentDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        mvReferenceDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        mvCurrentIntensity[lvl].create(hLvl, wLvl, CV_32FC1);
        mvReferenceIntensity[lvl].create(hLvl, wLvl, CV_32FC1);
        mvIntensityGradientX[lvl].create(hLvl, wLvl, CV_32FC1);
        mvIntensityGradientY[lvl].create(hLvl, wLvl, CV_32FC1);
        mvReferencePointTransformed[lvl].create(hLvl, wLvl, CV_32FC4);

        mvCurrentInvDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        mvReferenceInvDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        mvInvDepthGradientX[lvl].create(hLvl, wLvl, CV_32FC1);
        mvInvDepthGradientY[lvl].create(hLvl, wLvl, CV_32FC1);
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

void RGBDTracking::SetReferenceImage(const cv::Mat &imGray)
{
    cv::Mat imGrayFloat;
    imGray.convertTo(imGrayFloat, CV_32FC1);

    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        std::cout << mvReferenceIntensity[lvl].size() << std::endl;
        if (lvl == 0)
            mvReferenceIntensity[0].upload(imGrayFloat);
        else
            cv::cuda::pyrDown(mvReferenceIntensity[lvl - 1], mvReferenceIntensity[lvl]);
    }
}

void RGBDTracking::SetReferenceDepth(const cv::Mat &imDepth)
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
    }
}

void RGBDTracking::SetTrackingImage(const cv::Mat &imGray)
{
    cv::Mat imGrayFloat;
    imGray.convertTo(imGrayFloat, CV_32FC1);

    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        if (lvl == 0)
            mvCurrentIntensity[lvl].upload(imGrayFloat);
        else
            cv::cuda::pyrDown(mvCurrentIntensity[lvl - 1], mvCurrentIntensity[lvl]);

        ComputeImageGradientCentralDifference(mvCurrentIntensity[lvl], mvIntensityGradientX[lvl], mvIntensityGradientY[lvl]);
    }
}

void RGBDTracking::SetTrackingDepth(const cv::Mat &imDepth)
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

        ComputeImageGradientCentralDifference(mvCurrentInvDepth[lvl], mvInvDepthGradientX[lvl], mvInvDepthGradientY[lvl]);
    }
}

void RGBDTracking::SetReferenceInvD(cv::cuda::GpuMat vmap)
{
}

Sophus::SE3d RGBDTracking::GetTransform(const Sophus::SE3d &init, const bool bSwapBuffer)
{
    int nIteration = 0;
    int nSuccessfulIteration = 0;

    Sophus::SE3d estimate = init;
    Sophus::SE3d lastSuccessEstimate = estimate;
    std::vector<int> vIterations = {10, 5, 3, 3, 3};

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
                ComputeSingleStepRGBD(lvl, estimate, hessian.data(), residual.data());
                break;
            }

            float error = sqrt(residualSum) / (numResidual + 1);
            Eigen::Matrix<double, 6, 1> update = hessian.cast<double>().ldlt().solve(residual.cast<double>());

            if (std::isnan(update(0)))
            {
                mbTrackingGood = false;
                return Sophus::SE3d();
            }

            // update = ClampEigenVector(update, 0.05, -0.05);

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
        for (int lvl = 0; lvl < NUM_PYR; ++lvl)
        {
            std::swap(mvReferenceInvDepth[lvl], mvCurrentInvDepth[lvl]);
            std::swap(mvReferenceIntensity[lvl], mvCurrentIntensity[lvl]);
        }
    }

    mbTrackingGood = true;
    return lastSuccessEstimate;
}

void RGBDTracking::TransformReferencePoint(const int lvl, const Sophus::SE3d &T)
{
    auto refInvDepth = mvReferenceInvDepth[lvl];
    auto refPtTransformedLvl = mvReferencePointTransformed[lvl];
    auto KLvl = mK[lvl];

    ::TransformReferencePoint(refInvDepth, refPtTransformedLvl, KLvl, T);
}

void RGBDTracking::ComputeSingleStepRGB(
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

void RGBDTracking::SwapFrameBuffer()
{
    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        std::swap(mvReferenceInvDepth[lvl], mvCurrentInvDepth[lvl]);
        std::swap(mvReferenceIntensity[lvl], mvCurrentIntensity[lvl]);
    }
}

void RGBDTracking::ComputeSingleStepDepth(
    const int lvl,
    const Sophus::SE3d &T,
    float *hessian,
    float *residual)
{
    TransformReferencePoint(lvl, T);

    const int w = mvWidth[lvl];
    const int h = mvHeight[lvl];

    se3StepDResidualFunctor functor;
    functor.w = w;
    functor.h = h;
    functor.n = w * h;
    functor.refInvDepth = mvReferenceInvDepth[lvl];
    functor.currInvDepth = mvCurrentInvDepth[lvl];
    functor.currIDepthGx = mvInvDepthGradientX[lvl];
    functor.currIDepthGy = mvInvDepthGradientY[lvl];
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

    dResidualSum = hostData.ptr<float>(0)[0];
    numResidual = hostData.ptr<float>(0)[1];

    VarianceEstimator estimator;
    estimator.w = w;
    estimator.h = h;
    estimator.n = w * h;
    estimator.meanEstimated = dResidualSum / numResidual;
    estimator.residual = mGpuBufferVector4HxW;
    estimator.out = mGpuBufferFloat96x1;

    callDeviceFunctor<<<96, 224>>>(estimator);
    cv::cuda::reduce(mGpuBufferFloat96x1, mGpuBufferFloat1x1, 0, cv::REDUCE_SUM);
    mGpuBufferFloat1x1.download(hostData);

    float squaredDeviationSum = hostData.ptr<float>(0)[0];
    float varEstimated = sqrt(squaredDeviationSum / (numResidual - 1));

    se3StepDFunctor sfunctor;
    sfunctor.w = w;
    sfunctor.h = h;
    sfunctor.n = w * h;
    sfunctor.huberTh = 1.345 * varEstimated;
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

void RGBDTracking::ComputeSingleStepRGBD(
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

void RGBDTracking::ComputeSingleStepRGBDLinear(
    const int lvl,
    const Sophus::SE3d &T,
    float *hessian,
    float *residual)
{
    Eigen::Map<Eigen::Matrix<float, 6, 6>> hessianMapped(hessian);
    Eigen::Map<Eigen::Matrix<float, 6, 1>> residualMapped(residual);

    Eigen::Matrix<float, 6, 6> hessianBuffer;
    Eigen::Matrix<float, 6, 1> residualBuffer;

    ComputeSingleStepRGB(
        lvl,
        T,
        hessianBuffer.data(),
        residualBuffer.data());

    hessianMapped += 0.001 * hessianBuffer;
    residualMapped += 0.001 * residualBuffer;

    ComputeSingleStepDepth(
        lvl,
        T,
        hessianBuffer.data(),
        residualBuffer.data());

    hessianMapped += hessianBuffer;
    residualMapped += residualBuffer;
}

cv::cuda::GpuMat RGBDTracking::GetReferenceDepth(const int lvl) const
{
    return mGpuBufferRawDepth;
}

bool RGBDTracking::IsTrackingGood() const
{
    return mbTrackingGood;
}
