#include "DenseTracking.h"
#include "se3StepFunctor.h"
#include "ImageProc.h"
#include "statisticsFunctor.h"

DenseTracking::DenseTracking(const int &imgWidth, const int &imgHeight,
                             const Eigen::Matrix3d &K, const int &nPyrLvl,
                             const std::vector<int> &vIterations,
                             const bool &bUseRGB, const bool &bUseDepth)
    : mnNumPyr(nPyrLvl), mbTrackingGood(false), mvIterations(vIterations)
{
    if (!(bUseRGB || bUseDepth))
    {
        std::cout << "must choose a tracking modality." << std::endl;
        std::cout << "Dense Tracking will do NOTHING for now." << std::endl;
        mModal = TrackingModal::IDLE;
    }

    if (bUseRGB)
    {
        if (bUseDepth)
            mModal = TrackingModal::RGB_AND_DEPTH;
        else
            mModal = TrackingModal::RGB_ONLY;
    }
    else
        mModal = TrackingModal::DEPTH_ONLY;

    // Generate camera pyramid
    mvImageWidth.resize(nPyrLvl, -1);
    mvImageHeight.resize(nPyrLvl, -1);
    mK.resize(nPyrLvl);

    for (int lvl = 0; lvl < nPyrLvl; ++lvl)
    {
        int wLvl = imgWidth / (1 << lvl);
        int hLvl = imgHeight / (1 << lvl);

        mvImageWidth[lvl] = wLvl;
        mvImageHeight[lvl] = hLvl;

        mK[lvl] = K / (1 << lvl);
        mK[lvl](2, 2) = 1.0f;
    }

    // Allocate GPU buffers
    mvCurrentDepth.resize(nPyrLvl);
    mvReferenceDepth.resize(nPyrLvl);
    mvCurrentIntensity.resize(nPyrLvl);
    mvReferenceIntensity.resize(nPyrLvl);
    mvIntensityGradientX.resize(nPyrLvl);
    mvIntensityGradientY.resize(nPyrLvl);
    mvReferencePointTransformed.resize(nPyrLvl);
    mvCurrentInvDepth.resize(nPyrLvl);
    mvReferenceInvDepth.resize(nPyrLvl);
    mvInvDepthGradientX.resize(nPyrLvl);
    mvInvDepthGradientY.resize(nPyrLvl);

    for (int lvl = 0; lvl < nPyrLvl; ++lvl)
    {
        int wLvl = mvImageWidth[lvl];
        int hLvl = mvImageHeight[lvl];

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
    mGpuBufferVector4HxW.create(imgHeight, imgWidth, CV_32FC4);
    mGpuBufferVector7HxW.create(imgHeight, imgWidth, CV_32FC(7));
    mGpuBufferRawDepth.create(imgHeight, imgWidth, CV_32FC1);
}

void DenseTracking::SetReferenceImage(const cv::Mat &imGray)
{
    cv::Mat imGrayFloat;
    imGray.convertTo(imGrayFloat, CV_32FC1);

    for (int lvl = 0; lvl < mnNumPyr; ++lvl)
    {
        if (lvl == 0)
            mvReferenceIntensity[0].upload(imGrayFloat);
        else
            cv::cuda::pyrDown(mvReferenceIntensity[lvl - 1], mvReferenceIntensity[lvl]);
    }
}

void DenseTracking::SetReferenceDepth(const cv::Mat &imDepth)
{
    for (int lvl = 0; lvl < mnNumPyr; ++lvl)
    {
        if (lvl == 0)
        {
            mGpuBufferRawDepth.upload(imDepth);
            ImageProc::convertDepthToInvDepth(mGpuBufferRawDepth, mvReferenceInvDepth[lvl]);
        }
        else
            ImageProc::pyrdownInvDepth(mvReferenceInvDepth[lvl - 1], mvReferenceInvDepth[lvl]);
    }
}

void DenseTracking::SetTrackingImage(const cv::Mat &imGray)
{
    cv::Mat imGrayFloat;
    imGray.convertTo(imGrayFloat, CV_32FC1);

    for (int lvl = 0; lvl < mnNumPyr; ++lvl)
    {
        if (lvl == 0)
            mvCurrentIntensity[lvl].upload(imGrayFloat);
        else
            cv::cuda::pyrDown(mvCurrentIntensity[lvl - 1], mvCurrentIntensity[lvl]);

        ImageProc::computeImageGradientCentralDiff(mvCurrentIntensity[lvl], mvIntensityGradientX[lvl], mvIntensityGradientY[lvl]);
    }
}

void DenseTracking::SetTrackingDepth(const cv::Mat &imDepth)
{
    for (int lvl = 0; lvl < mnNumPyr; ++lvl)
    {
        if (lvl == 0)
        {
            mGpuBufferRawDepth.upload(imDepth);
            ImageProc::convertDepthToInvDepth(mGpuBufferRawDepth, mvCurrentInvDepth[lvl]);
        }
        else
            ImageProc::pyrdownInvDepth(mvCurrentInvDepth[lvl - 1], mvCurrentInvDepth[lvl]);

        ImageProc::computeImageGradientCentralDiff(mvCurrentInvDepth[lvl], mvInvDepthGradientX[lvl], mvInvDepthGradientY[lvl]);
    }
}

void DenseTracking::SetReferenceInvD(cv::cuda::GpuMat vmap)
{
    for (int lvl = 0; lvl < mnNumPyr; ++lvl)
    {
        if (lvl == 0)
            ImageProc::convertVMapToInvDepth(vmap, mvReferenceInvDepth[lvl]);
        else
            cv::cuda::pyrDown(mvReferenceInvDepth[lvl - 1], mvReferenceInvDepth[lvl]);
    }
}

Sophus::SE3d DenseTracking::GetTransform(const Sophus::SE3d &init, const bool bSwapBuffer)
{
    Sophus::SE3d estimate = init;
    Sophus::SE3d lastSuccessEstimate = estimate;
    for (int lvl = mnNumPyr - 1; lvl >= 0; --lvl)
    {
        float lastError = std::numeric_limits<float>::max();

        for (int iter = 0; iter < mvIterations[lvl]; ++iter)
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

            case TrackingModal::IDLE:
                break;
            }

            float error = sqrt(residualSum) / (numResidual + 1);
            Eigen::Matrix<double, 6, 1> update = hessian.cast<double>().ldlt().solve(residual.cast<double>());

            if (std::isnan(update(0)))
            {
                mbTrackingGood = false;
                return Sophus::SE3d();
            }

            estimate = Sophus::SE3d::exp(update) * estimate;
            if (error < lastError)
            {
                lastSuccessEstimate = estimate;
                lastError = error;
            }
        }
    }

    if (bSwapBuffer)
    {
        for (int lvl = 0; lvl < mnNumPyr; ++lvl)
        {
            std::swap(mvReferenceInvDepth[lvl], mvCurrentInvDepth[lvl]);
            std::swap(mvReferenceIntensity[lvl], mvCurrentIntensity[lvl]);
        }
    }

    mbTrackingGood = true;
    return lastSuccessEstimate;
}

void DenseTracking::TransformReferencePoint(const int lvl, const Sophus::SE3d &T)
{
    auto refInvDepth = mvReferenceInvDepth[lvl];
    auto refPtTransformedLvl = mvReferencePointTransformed[lvl];
    auto KLvl = mK[lvl];

    ImageProc::TransformReferencePoint(refInvDepth, refPtTransformedLvl, KLvl, T);
}

void DenseTracking::ComputeSingleStepRGB(
    const int lvl,
    const Sophus::SE3d &T,
    float *hessian,
    float *residual)
{
    TransformReferencePoint(lvl, T);

    const int w = mvImageWidth[lvl];
    const int h = mvImageHeight[lvl];

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

void DenseTracking::SwapFrameBuffer()
{
    for (int lvl = 0; lvl < mnNumPyr; ++lvl)
    {
        std::swap(mvReferenceInvDepth[lvl], mvCurrentInvDepth[lvl]);
        std::swap(mvReferenceIntensity[lvl], mvCurrentIntensity[lvl]);
    }
}

void DenseTracking::ComputeSingleStepDepth(
    const int lvl,
    const Sophus::SE3d &T,
    float *hessian,
    float *residual)
{
    TransformReferencePoint(lvl, T);

    const int w = mvImageWidth[lvl];
    const int h = mvImageHeight[lvl];

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

void DenseTracking::ComputeSingleStepRGBD(
    const int lvl,
    const Sophus::SE3d &T,
    float *hessian,
    float *residual)
{
    TransformReferencePoint(lvl, T);

    const int w = mvImageWidth[lvl];
    const int h = mvImageHeight[lvl];

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

void DenseTracking::ComputeSingleStepRGBDLinear(
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

cv::cuda::GpuMat DenseTracking::GetReferenceDepth(const int lvl) const
{
    return mGpuBufferRawDepth;
}

bool DenseTracking::IsTrackingGood() const
{
    return mbTrackingGood;
}