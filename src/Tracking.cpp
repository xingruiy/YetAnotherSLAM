#include "Tracking.h"
#include "ImageProc.h"

namespace SLAM
{

Tracking::Tracking(System *system, Map *map, Viewer *viewer, Mapping *mapping)
    : slamSystem(system), mpMap(map), viewer(viewer), mapping(mapping), trackingState(Null)
{
    mapper = new DenseMapping(g_width[0], g_height[0], g_calib[0].cast<double>());
    tracker = new DenseTracking(g_width[0], g_height[0], g_calib[0].cast<double>(), NUM_PYR, {10, 5, 3, 3, 3}, g_bUseColour, g_bUseDepth);
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
    // mapper->fuseFrame(cv::cuda::GpuMat(NextFrame.mImDepth), NextFrame.mTcw, (uint)NextFrame.mnId);

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

    // mapper->fuseFrame(cv::cuda::GpuMat(NextFrame.mImDepth), NextFrame.mTcw, (uint)NextFrame.mnId);
    // mapper->raytrace(NextFrame.mTcw);
    // tracker->SetReferenceInvD(mapper->GetSyntheticVertexMap());

    return true;
}

bool Tracking::relocalisation()
{
    return false;
}

bool Tracking::NeedNewKeyFrame()
{

    Sophus::SE3d DT = T_ref2World.inverse() * NextFrame.mTcw;

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

    mapping->AddKeyFrameCandidate(NextFrame);

    // Set to the reference frame
    NextFrame.T_frame2Ref = Sophus::SE3d();
}

void Tracking::reset()
{
    trackingState = Null;
}

} // namespace SLAM