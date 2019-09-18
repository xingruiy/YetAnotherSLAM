#include "denseTracker/denseTracker.h"
#include "denseTracker/se3StepFunctor.h"
#include "denseTracker/cudaImageProc.h"
#include "denseTracker/statisticsFunctor.h"

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
    }

    iterationPerLvl = {10, 5, 4, 3, 3};
}

void DenseTracker::setReferenceInvDepth(GMat ref)
{
}

void DenseTracker::setReferenceFrame(std::shared_ptr<Frame> frame)
{
    auto depth = frame->getDepth();
    auto image = frame->getImage();

    referenceDepth[0].upload(depth);
    referenceIntensity[0].upload(image);

    for (int lvl = 1; lvl < numTrackingLvl; ++lvl)
    {
        cv::cuda::pyrDown(referenceDepth[lvl - 1], referenceDepth[lvl]);
        cv::cuda::pyrDown(referenceIntensity[lvl - 1], referenceIntensity[lvl]);
    }
}

void DenseTracker::setTrackingFrame(std::shared_ptr<Frame> frame)
{
    auto depth = frame->getDepth();
    auto image = frame->getImage();
    currentDepth[0].upload(depth);
    currentIntensity[0].upload(image);

    for (int lvl = 1; lvl < numTrackingLvl; ++lvl)
    {
        cv::cuda::pyrDown(currentDepth[lvl - 1], currentDepth[lvl]);
        cv::cuda::pyrDown(currentIntensity[lvl - 1], currentIntensity[lvl]);
    }

    for (int lvl = 0; lvl < numTrackingLvl; ++lvl)
        computeImageGradientCentralDiff(currentIntensity[lvl], intensityGradientX[lvl], intensityGradientY[lvl]);
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

            computeSE3StepRGB(
                lvl,
                estimate,
                hessian.data(),
                residual.data());

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
            std::swap(referenceDepth[lvl], currentDepth[lvl]);
            std::swap(referenceIntensity[lvl], currentIntensity[lvl]);
        }

    return lastSuccessEstimate;
}

void DenseTracker::transformReferencePoint(const int lvl, const SE3 &estimate)
{
    auto referenceDepthLvl = referenceDepth[lvl];
    auto referencePointTransformedLvl = referencePointTransformed[lvl];
    auto KLvl = intrinsics[lvl];

    ::transformReferencePoint(referenceDepthLvl, referencePointTransformedLvl, KLvl, estimate);
}

void DenseTracker::computeSE3StepRGB(
    const int lvl,
    SE3 &estimate,
    float *hessian,
    float *residual)
{
    transformReferencePoint(lvl, estimate);

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

GMat DenseTracker::getReferenceDepth(const int lvl) const
{
    return referenceDepth[lvl];
}