#include "Tracking.h"
#include "ImageProc.h"

namespace SLAM
{

Tracking::Tracking(System *system, Map *map, Viewer *viewer, Mapping *mapping)
    : slamSystem(system), mpMap(map), viewer(viewer), mapping(mapping),
      trackingState(Null)
{
    tracker = new DenseTracking(g_width[0], g_height[0], g_calib[0].cast<double>(), NUM_PYR, {10, 5, 3, 3, 3}, g_bUseColour, g_bUseDepth);

    if (g_bUseDepth && g_bUseColour)
    {
        trackingModal = TrackingModal::RGB_AND_DEPTH;
    }
    else if (g_bUseColour)
    {
        trackingModal = TrackingModal::RGB_ONLY;
    }
    else if (g_bUseDepth)
    {
        trackingModal = TrackingModal::DEPTH_ONLY;
    }
    else
    {
        std::cout << "You must choose a tracking modality." << std::endl;
        std::cout << "Otherwise the system will do NOTHING." << std::endl;
        trackingModal = TrackingModal::IDLE;
    }

    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        int wLvl = g_width[lvl];
        int hLvl = g_height[lvl];

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
    mGpuBufferVector4HxW.create(g_height[0], g_width[0], CV_32FC4);
    mGpuBufferVector7HxW.create(g_height[0], g_width[0], CV_32FC(7));
    mGpuBufferRawDepth.create(g_height[0], g_width[0], CV_32FC1);

    mvIterations = {10, 5, 3, 3, 3};
}

void Tracking::trackImage(cv::Mat ImgGray, cv::Mat ImgDepth, const double TimeStamp)
{
    NextFrame = Frame(ImgGray, ImgDepth, TimeStamp);

    bool bOK = false;
    switch (trackingState)
    {
    case Null:
    {
        initialisation();
        break;
    }

    case OK:
    {
        bool bOK = trackLastFrame();

        if (bOK)
        {
            if (NeedNewKeyFrame())
                MakeNewKeyFrame();
        }
        else
        {
            trackingState = Lost;
        }

        break;
    }

    case Lost:
    {
        bool bOK = relocalisation();

        if (bOK)
        {
            trackingState = OK;
            break;
        }
        else
            return;
    }
    }

    lastFrame = Frame(NextFrame);
}

void Tracking::initialisation()
{
    tracker->SetReferenceImage(NextFrame.mImGray);
    tracker->SetReferenceDepth(NextFrame.mImDepth);

    T_ref2World = NextFrame.mTcw;
    NextFrame.mTcw = Sophus::SE3d(Eigen::Matrix4d::Identity());
    NextFrame.T_frame2Ref = Sophus::SE3d(Eigen::Matrix4d::Identity());

    mapping->AddKeyFrameCandidate(NextFrame);

    trackingState = OK;
}

bool Tracking::trackLastFrame()
{
    tracker->SetTrackingImage(NextFrame.mImGray);
    tracker->SetTrackingDepth(NextFrame.mImDepth);

    Sophus::SE3d Tpc = tracker->GetTransform();

    NextFrame.mTcw = lastFrame.mTcw * Tpc.inverse();
    NextFrame.T_frame2Ref = lastFrame.T_frame2Ref * Tpc.inverse();

    if (g_bEnableViewer)
        viewer->setLivePose(NextFrame.mTcw.matrix());

    return true;
}

bool Tracking::relocalisation()
{
    return false;
}

bool Tracking::NeedNewKeyFrame()
{

    Sophus::SE3d DT = T_ref2World.inverse() * NextFrame.mTcw;

    if (DT.log().topRows<3>().norm() > 0.3)
        return true;

    if (DT.log().bottomRows<3>().norm() > 0.3)
        return true;

    return false;
}

void Tracking::MakeNewKeyFrame()
{
    // Update keyframe pose
    T_ref2World = NextFrame.mTcw;

    mapping->AddKeyFrameCandidate(NextFrame);

    // Set to the reference frame
    NextFrame.T_frame2Ref = Sophus::SE3d();
}

void Tracking::reset()
{
    trackingState = Null;
}

void Tracking::SetReferenceFrame(const Frame &F)
{
    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        if (lvl == 0)
        {
            mvReferenceIntensity[0].upload(F.mImGray);
            mGpuBufferRawDepth.upload(F.mImDepth);
            ImageProc::convertDepthToInvDepth(mGpuBufferRawDepth, mvReferenceInvDepth[lvl]);
        }
        else
        {
            cv::cuda::pyrDown(mvReferenceIntensity[lvl - 1], mvReferenceIntensity[lvl]);
            ImageProc::pyrdownInvDepth(mvReferenceInvDepth[lvl - 1], mvReferenceInvDepth[lvl]);
        }
    }
}

void Tracking::SetNextFrame(const Frame &F)
{
    cv::Mat imGrayFloat;
    F.mImGray.convertTo(imGrayFloat, CV_32FC1);

    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        if (lvl == 0)
        {
            mvCurrentIntensity[lvl].upload(imGrayFloat);
            mGpuBufferRawDepth.upload(F.mImDepth);
            ImageProc::convertDepthToInvDepth(mGpuBufferRawDepth, mvCurrentInvDepth[lvl]);
        }
        else
        {
            cv::cuda::pyrDown(mvCurrentIntensity[lvl - 1], mvCurrentIntensity[lvl]);
            ImageProc::pyrdownInvDepth(mvCurrentInvDepth[lvl - 1], mvCurrentInvDepth[lvl]);
        }

        ImageProc::computeImageGradientCentralDiff(mvCurrentIntensity[lvl], mvIntensityGradientX[lvl], mvIntensityGradientY[lvl]);
        ImageProc::computeImageGradientCentralDiff(mvCurrentInvDepth[lvl], mvInvDepthGradientX[lvl], mvInvDepthGradientY[lvl]);
    }
}

Sophus::SE3d Tracking::ComputeCoarseTransform()
{
    Sophus::SE3d estimate = Sophus::SE3d();
    Sophus::SE3d lastSuccessEstimate = estimate;
    for (int lvl = NUM_PYR - 1; lvl >= 0; --lvl)
    {
        float lastError = std::numeric_limits<float>::max();
        for (int iter = 0; iter < mvIterations[lvl]; ++iter)
        {
            Eigen::Matrix<float, 6, 6> hessian = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Matrix<float, 6, 1> residual = Eigen::Matrix<float, 6, 1>::Zero();

            switch (trackingModal)
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

    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        std::swap(mvReferenceInvDepth[lvl], mvCurrentInvDepth[lvl]);
        std::swap(mvReferenceIntensity[lvl], mvCurrentIntensity[lvl]);
    }

    mbTrackingGood = true;
    return lastSuccessEstimate;
}

void Tracking::TransformReferencePoint(const int lvl, const Sophus::SE3d &T)
{
    auto refInvDepth = mvReferenceInvDepth[lvl];
    auto refPtTransformedLvl = mvReferencePointTransformed[lvl];
    auto KLvl = g_calib[lvl].cast<double>();

    ImageProc::TransformReferencePoint(refInvDepth, refPtTransformedLvl, KLvl, T);
}

void Tracking::ComputeSingleStepRGB(const int lvl, const Sophus::SE3d &T, float *hessian, float *residual)
{
}

void Tracking::ComputeSingleStepRGBDLinear(const int lvl, const Sophus::SE3d &T, float *hessian, float *residual)
{
}

void Tracking::ComputeSingleStepRGBD(const int lvl, const Sophus::SE3d &T, float *hessian, float *residual)
{
}

void Tracking::ComputeSingleStepDepth(const int lvl, const Sophus::SE3d &T, float *hessian, float *residual)
{
}

} // namespace SLAM