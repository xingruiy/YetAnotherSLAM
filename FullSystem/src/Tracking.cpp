#include "Tracking.h"

namespace SLAM
{

Tracking::Tracking(System *system, Map *map, Viewer *mpViewer, LocalMapping *mpLocalMapping)
    : mpSystem(system), mpMap(map), mpViewer(mpViewer), mpLocalMapping(mpLocalMapping), trackingState(Null)
{
    int w = g_width[0];
    int h = g_height[0];
    Eigen::Matrix3f calib = g_calib[0];

    mpTracker = new DenseTracking(w, h, calib.cast<double>(), NUM_PYR, {10, 5, 3, 3, 3}, g_bUseColour, g_bUseDepth);
    // mpLocalMapper = new DenseMapping(w, h, calib);
}

void Tracking::trackImage(cv::Mat ImgGray, cv::Mat ImgDepth, const double TimeStamp)
{
    NextFrame = Frame(ImgGray, ImgDepth, TimeStamp);

    bool bOK = false;
    switch (trackingState)
    {
    case Null:
    {
        Initialisation();
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
        bool bOK = Relocalisation();

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

void Tracking::Initialisation()
{
    mpTracker->SetReferenceImage(NextFrame.mImGray);
    mpTracker->SetReferenceDepth(NextFrame.mImDepth);
    // mpLocalMapper->fuseFrame(cv::cuda::GpuMat(NextFrame.mImDepth), NextFrame.mTcw);

    T_ref2World = NextFrame.mTcw;
    NextFrame.mTcw = Sophus::SE3d(Eigen::Matrix4d::Identity());
    NextFrame.T_frame2Ref = Sophus::SE3d(Eigen::Matrix4d::Identity());

    mpLocalMapping->AddKeyFrameCandidate(NextFrame);

    trackingState = OK;
}

bool Tracking::trackLastFrame()
{
    mpTracker->SetTrackingImage(NextFrame.mImGray);
    mpTracker->SetTrackingDepth(NextFrame.mImDepth);

    Sophus::SE3d Tpc = mpTracker->GetTransform(lastFrame.T_frame2Ref.inverse(), false);

    // NextFrame.mTcw = lastFrame.mTcw * Tpc.inverse();
    // NextFrame.T_frame2Ref = lastFrame.T_frame2Ref * Tpc.inverse();
    NextFrame.mTcw = T_ref2World * Tpc.inverse();
    NextFrame.T_frame2Ref = Tpc.inverse();

    if (g_bEnableViewer)
        mpViewer->setLivePose(NextFrame.mTcw.matrix());

    // cv::cuda::GpuMat vmap(480, 640, CV_32FC4);
    // mpLocalMapper->fuseFrame(cv::cuda::GpuMat(NextFrame.mImDepth), NextFrame.mTcw);
    // mpLocalMapper->raytrace(vmap, NextFrame.mTcw);
    // cv::imshow("vmap", cv::Mat(vmap));
    // cv::waitKey(1);

    g_nTrackedFrame++;
    return true;
}

bool Tracking::Relocalisation()
{
    return false;
}

bool Tracking::NeedNewKeyFrame()
{

    // Sophus::SE3d DT = T_ref2World.inverse() * NextFrame.mTcw;
    Sophus::SE3d DT = NextFrame.T_frame2Ref;

    if (DT.log().topRows<3>().norm() > 0.15)
        return true;

    if (DT.log().bottomRows<3>().norm() > 0.2)
        return true;

    return false;
}

void Tracking::MakeNewKeyFrame()
{
    // Update keyframe pose
    T_ref2World = NextFrame.mTcw;

    mpLocalMapping->AddKeyFrameCandidate(NextFrame);
    mpTracker->SwapFrameBuffer();

    // Set to the reference frame
    NextFrame.T_frame2Ref = Sophus::SE3d();
}

void Tracking::reset()
{
    trackingState = Null;
}

} // namespace SLAM