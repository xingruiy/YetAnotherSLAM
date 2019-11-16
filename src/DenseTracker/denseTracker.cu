#include "DenseTracker/denseTracker.h"
#include "DenseTracker/se3StepFunctor.h"
#include "DenseTracker/cudaImageProc.h"
#include "DenseTracker/statisticsFunctor.h"

DenseTracker::DenseTracker(int w, int h, Mat33d &K, int numLvl)
    : numTrackingLvl(numLvl)
{
    bufferFloat96x29.create(96, 29, CV_32FC1);
    bufferFloat96x3.create(96, 3, CV_32FC1);
    bufferFloat96x2.create(96, 2, CV_32FC1);
    bufferFloat96x1.create(96, 1, CV_32FC1);
    bufferFloat1x29.create(1, 29, CV_32FC1);
    bufferFloat1x3.create(1, 2, CV_32FC1);
    bufferFloat1x2.create(1, 2, CV_32FC1);
    bufferFloat1x1.create(1, 1, CV_32FC1);
    bufferVec4hxw.create(h, w, CV_32FC4);
    bufferVec7hxw.create(h, w, CV_32FC7);

    frameWidth.resize(numLvl);
    frameHeight.resize(numLvl);
    intrinsics.resize(numLvl);

    currentDepth.resize(numLvl);
    referenceDepth.resize(numLvl);
    currentIntensity.resize(numLvl);
    referenceIntensity.resize(numLvl);
    intensityGradientX.resize(numLvl);
    intensityGradientY.resize(numLvl);
    referencePointTransformed.resize(numLvl);

    currentInvDepth.resize(numLvl);
    referenceInvDepth.resize(numLvl);
    invDepthGradientX.resize(numLvl);
    invDepthGradientY.resize(numLvl);

    rawImageBuffer.create(h, w, CV_8UC3);
    rawDepthBuffer.create(h, w, CV_32FC1);

    for (int lvl = 0; lvl < numLvl; ++lvl)
    {
        int wLvl = w / (1 << lvl);
        int hLvl = h / (1 << lvl);

        frameWidth[lvl] = wLvl;
        frameHeight[lvl] = hLvl;
        intrinsics[lvl] = K / (1 << lvl);
        intrinsics[lvl](2, 2) = 1.0f;

        currentDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        referenceDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        currentIntensity[lvl].create(hLvl, wLvl, CV_32FC1);
        referenceIntensity[lvl].create(hLvl, wLvl, CV_32FC1);
        intensityGradientX[lvl].create(hLvl, wLvl, CV_32FC1);
        intensityGradientY[lvl].create(hLvl, wLvl, CV_32FC1);
        referencePointTransformed[lvl].create(hLvl, wLvl, CV_32FC4);

        currentInvDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        referenceInvDepth[lvl].create(hLvl, wLvl, CV_32FC1);
        invDepthGradientX[lvl].create(hLvl, wLvl, CV_32FC1);
        invDepthGradientY[lvl].create(hLvl, wLvl, CV_32FC1);
    }

    iterationPerLvl = {10, 5, 4, 3, 3};
}

void DenseTracker::setReferenceInvDepth(GMat vmap)
{
    for (int lvl = 1; lvl < numTrackingLvl; ++lvl)
    {
        if (lvl == 0)
            convertVMapToInvDepth(vmap, referenceInvDepth[lvl]);
        else
            cv::cuda::pyrDown(referenceInvDepth[lvl - 1], referenceInvDepth[lvl]);
    }
}

void DenseTracker::setReferenceFrame(const Frame &F)
{
    for (int lvl = 0; lvl < numTrackingLvl; ++lvl)
    {
        if (lvl == 0)
        {
            rawDepthBuffer.upload(F.imDepth);
            convertDepthToInvDepth(rawDepthBuffer, referenceInvDepth[lvl]);
            referenceIntensity[0].upload(F.imGray);
        }
        else
        {
            pyrdownInvDepth(referenceInvDepth[lvl - 1], referenceInvDepth[lvl]);
            cv::cuda::pyrDown(referenceIntensity[lvl - 1], referenceIntensity[lvl]);
        }
    }
}

void DenseTracker::setTrackingFrame(const Frame &F)
{
    for (int lvl = 0; lvl < numTrackingLvl; ++lvl)
    {
        if (lvl == 0)
        {
            rawDepthBuffer.upload(F.imDepth);
            convertDepthToInvDepth(rawDepthBuffer, currentInvDepth[lvl]);
            currentIntensity[lvl].upload(F.imGray);
        }
        else
        {
            pyrdownInvDepth(currentInvDepth[lvl - 1], currentInvDepth[lvl]);
            cv::cuda::pyrDown(currentIntensity[lvl - 1], currentIntensity[lvl]);
        }

        computeImageGradientCentralDiff(currentInvDepth[lvl], invDepthGradientX[lvl], invDepthGradientY[lvl]);
        computeImageGradientCentralDiff(currentIntensity[lvl], intensityGradientX[lvl], intensityGradientY[lvl]);
    }
}

SE3 DenseTracker::getIncrementalTransform(SE3 initAlign, bool switchBuffer)
{
    SE3 estimate = initAlign;
    SE3 lastSuccessEstimate = estimate;

    for (int lvl = numTrackingLvl - 1; lvl >= 0; --lvl)
    {
        float lastError = std::numeric_limits<float>::max();

        for (int iter = 0; iter < iterationPerLvl[lvl]; ++iter)
        {
            Mat66f hessian = Mat66f::Zero();
            Vec6f residual = Vec6f::Zero();

            // computeSE3StepRGB(
            //     lvl,
            //     estimate,
            //     hessian.data(),
            //     residual.data());

            // computeSE3StepD(
            //     lvl,
            //     estimate,
            //     hessian.data(),
            //     residual.data());

            computeSE3StepRGBD(
                lvl,
                estimate,
                hessian.data(),
                residual.data());

            // computeSE3StepRGBDLinear(
            //     lvl,
            //     estimate,
            //     hessian.data(),
            //     residual.data());

            float error = sqrt(residualSum) / (numResidual + 1);
            Vec6d update = hessian.cast<double>().ldlt().solve(residual.cast<double>());

            try
            {
                estimate = SE3::exp(update) * estimate;
                if (error < lastError)
                {
                    lastSuccessEstimate = estimate;
                    lastError = error;
                }
            }
            catch (std::exception e)
            {
                printf("Problems occured when computing pose transformation, verbose:\n%s", e.what());
                return SE3();
            }
        }
    }

    if (switchBuffer)
        for (int lvl = 0; lvl < numTrackingLvl; ++lvl)
        {
            // std::swap(referenceDepth[lvl], currentDepth[lvl]);
            std::swap(referenceInvDepth[lvl], currentInvDepth[lvl]);
            std::swap(referenceIntensity[lvl], currentIntensity[lvl]);
        }

    return lastSuccessEstimate;
}

void DenseTracker::transformReferencePoint(const int lvl, const SE3 &T)
{
    auto refInvDepth = referenceInvDepth[lvl];
    auto refPtTransformedLvl = referencePointTransformed[lvl];
    auto KLvl = intrinsics[lvl];

    ::transformReferencePoint(refInvDepth, refPtTransformedLvl, KLvl, T);
}

void DenseTracker::computeSE3StepRGB(
    const int lvl,
    const SE3 &T,
    float *hessian,
    float *residual)
{
    transformReferencePoint(lvl, T);

    const int w = frameWidth[lvl];
    const int h = frameHeight[lvl];

    se3StepRGBResidualFunctor functor;
    functor.w = w;
    functor.h = h;
    functor.n = w * h;
    functor.refInt = referenceIntensity[lvl];
    functor.currInt = currentIntensity[lvl];
    functor.currGx = intensityGradientX[lvl];
    functor.currGy = intensityGradientY[lvl];
    functor.refPtWarped = referencePointTransformed[lvl];
    functor.refResidual = bufferVec4hxw;
    functor.fx = intrinsics[lvl](0, 0);
    functor.fy = intrinsics[lvl](1, 1);
    functor.cx = intrinsics[lvl](0, 2);
    functor.cy = intrinsics[lvl](1, 2);
    functor.out = bufferFloat96x2;

    callDeviceFunctor<<<96, 224>>>(functor);
    cv::cuda::reduce(bufferFloat96x2, bufferFloat1x2, 0, cv::REDUCE_SUM);
    cv::Mat hostData(bufferFloat1x2);

    iResidualSum = hostData.ptr<float>(0)[0];
    numResidual = hostData.ptr<float>(0)[1];

    VarianceEstimator estimator;
    estimator.w = w;
    estimator.h = h;
    estimator.n = w * h;
    estimator.meanEstimated = iResidualSum / numResidual;
    estimator.residual = bufferVec4hxw;
    estimator.out = bufferFloat96x1;

    callDeviceFunctor<<<96, 224>>>(estimator);
    cv::cuda::reduce(bufferFloat96x1, bufferFloat1x1, 0, cv::REDUCE_SUM);
    bufferFloat1x1.download(hostData);

    float squaredDeviationSum = hostData.ptr<float>(0)[0];
    float varEstimated = sqrt(squaredDeviationSum / (numResidual - 1));

    se3StepRGBFunctor sfunctor;
    sfunctor.w = w;
    sfunctor.h = h;
    sfunctor.n = w * h;
    sfunctor.huberTh = 4.685 * varEstimated;
    sfunctor.refPtWarped = referencePointTransformed[lvl];
    sfunctor.refResidual = bufferVec4hxw;
    sfunctor.fx = intrinsics[lvl](0, 0);
    sfunctor.fy = intrinsics[lvl](1, 1);
    sfunctor.out = bufferFloat96x29;

    callDeviceFunctor<<<96, 224>>>(sfunctor);
    cv::cuda::reduce(bufferFloat96x29, bufferFloat1x29, 0, cv::REDUCE_SUM);

    bufferFloat1x29.download(hostData);
    rankUpdateHessian<6, 7>(hostData.ptr<float>(0), hessian, residual);

    residualSum = hostData.ptr<float>(0)[27];
}

void DenseTracker::computeSE3StepD(
    const int lvl,
    const SE3 &T,
    float *hessian,
    float *residual)
{
    transformReferencePoint(lvl, T);

    const int w = frameWidth[lvl];
    const int h = frameHeight[lvl];

    se3StepDResidualFunctor functor;
    functor.w = w;
    functor.h = h;
    functor.n = w * h;
    functor.refInvDepth = referenceInvDepth[lvl];
    functor.currInvDepth = currentInvDepth[lvl];
    functor.currIDepthGx = invDepthGradientX[lvl];
    functor.currIDepthGy = invDepthGradientY[lvl];
    functor.refPtWarped = referencePointTransformed[lvl];
    functor.refResidual = bufferVec4hxw;
    functor.fx = intrinsics[lvl](0, 0);
    functor.fy = intrinsics[lvl](1, 1);
    functor.cx = intrinsics[lvl](0, 2);
    functor.cy = intrinsics[lvl](1, 2);
    functor.out = bufferFloat96x2;

    callDeviceFunctor<<<96, 224>>>(functor);
    cv::cuda::reduce(bufferFloat96x2, bufferFloat1x2, 0, cv::REDUCE_SUM);
    cv::Mat hostData(bufferFloat1x2);

    dResidualSum = hostData.ptr<float>(0)[0];
    numResidual = hostData.ptr<float>(0)[1];

    VarianceEstimator estimator;
    estimator.w = w;
    estimator.h = h;
    estimator.n = w * h;
    estimator.meanEstimated = dResidualSum / numResidual;
    estimator.residual = bufferVec4hxw;
    estimator.out = bufferFloat96x1;

    callDeviceFunctor<<<96, 224>>>(estimator);
    cv::cuda::reduce(bufferFloat96x1, bufferFloat1x1, 0, cv::REDUCE_SUM);
    bufferFloat1x1.download(hostData);

    float squaredDeviationSum = hostData.ptr<float>(0)[0];
    float varEstimated = sqrt(squaredDeviationSum / (numResidual - 1));

    se3StepDFunctor sfunctor;
    sfunctor.w = w;
    sfunctor.h = h;
    sfunctor.n = w * h;
    sfunctor.huberTh = 1.345 * varEstimated;
    sfunctor.refPtWarped = referencePointTransformed[lvl];
    sfunctor.refResidual = bufferVec4hxw;
    sfunctor.fx = intrinsics[lvl](0, 0);
    sfunctor.fy = intrinsics[lvl](1, 1);
    sfunctor.out = bufferFloat96x29;

    callDeviceFunctor<<<96, 224>>>(sfunctor);
    cv::cuda::reduce(bufferFloat96x29, bufferFloat1x29, 0, cv::REDUCE_SUM);

    bufferFloat1x29.download(hostData);
    rankUpdateHessian<6, 7>(hostData.ptr<float>(0), hessian, residual);

    residualSum = hostData.ptr<float>(0)[27];
}

void DenseTracker::computeSE3StepRGBD(
    const int lvl,
    const SE3 &T,
    float *hessian,
    float *residual)
{
    transformReferencePoint(lvl, T);

    const int w = frameWidth[lvl];
    const int h = frameHeight[lvl];

    se3StepRGBDResidualFunctor functor;
    functor.w = w;
    functor.h = h;
    functor.n = w * h;
    functor.refInt = referenceIntensity[lvl];
    functor.currInt = currentIntensity[lvl];
    functor.currGx = intensityGradientX[lvl];
    functor.currGy = intensityGradientY[lvl];
    functor.currInvDepth = currentInvDepth[lvl];
    functor.currInvDepthGx = invDepthGradientX[lvl];
    functor.currInvDepthGy = invDepthGradientY[lvl];
    functor.refPtWarped = referencePointTransformed[lvl];
    functor.refResidual = bufferVec7hxw;
    functor.fx = intrinsics[lvl](0, 0);
    functor.fy = intrinsics[lvl](1, 1);
    functor.cx = intrinsics[lvl](0, 2);
    functor.cy = intrinsics[lvl](1, 2);
    functor.out = bufferFloat96x3;

    callDeviceFunctor<<<96, 224>>>(functor);
    cv::cuda::reduce(bufferFloat96x3, bufferFloat1x3, 0, cv::REDUCE_SUM);
    cv::Mat hostData(bufferFloat1x3);

    float iResidualSum = hostData.ptr<float>(0)[0];
    float dResidualSum = hostData.ptr<float>(0)[1];
    numResidual = hostData.ptr<float>(0)[2];

    VarCov2DEstimator estimator;
    estimator.h = h;
    estimator.w = w;
    estimator.n = h * w;
    estimator.meanEstimated = Vec2f(iResidualSum, dResidualSum) / numResidual;
    estimator.residual = bufferVec7hxw;
    estimator.out = bufferFloat96x3;

    callDeviceFunctor<<<96, 224>>>(estimator);
    cv::cuda::reduce(bufferFloat96x3, bufferFloat1x3, 0, cv::REDUCE_SUM);
    bufferFloat1x3.download(hostData);

    Mat22f varEstimated;
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
    sfunctor.refPtWarped = referencePointTransformed[lvl];
    sfunctor.refResidual = bufferVec7hxw;
    sfunctor.fx = intrinsics[lvl](0, 0);
    sfunctor.fy = intrinsics[lvl](1, 1);
    sfunctor.out = bufferFloat96x29;

    callDeviceFunctor<<<96, 224>>>(sfunctor);
    cv::cuda::reduce(bufferFloat96x29, bufferFloat1x29, 0, cv::REDUCE_SUM);

    bufferFloat1x29.download(hostData);
    rankUpdateHessian<6, 7>(hostData.ptr<float>(0), hessian, residual);

    residualSum = hostData.ptr<float>(0)[27];
}

void DenseTracker::computeSE3StepRGBDLinear(
    const int lvl,
    const SE3 &T,
    float *hessian,
    float *residual)
{
    Eigen::Map<Mat66f> hessianMapped(hessian);
    Eigen::Map<Vec6f> residualMapped(residual);

    Mat66f hessianBuffer;
    Vec6f residualBuffer;

    computeSE3StepRGB(
        lvl,
        T,
        hessianBuffer.data(),
        residualBuffer.data());

    hessianMapped += 0.001 * hessianBuffer;
    residualMapped += 0.001 * residualBuffer;

    computeSE3StepD(
        lvl,
        T,
        hessianBuffer.data(),
        residualBuffer.data());

    hessianMapped += hessianBuffer;
    residualMapped += residualBuffer;
}

GMat DenseTracker::getReferenceDepth(const int lvl) const
{
    return rawDepthBuffer;
}