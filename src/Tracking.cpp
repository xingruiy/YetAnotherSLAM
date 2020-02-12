#include "Tracking.h"

namespace SLAM
{

Tracking::Tracking(System *system, Map *map, Viewer *viewer, Mapping *mapping)
    : slamSystem(system), mpMap(map), viewer(viewer), mapping(mapping),
      trackingState(Null), currentFrame(nullptr), lastFrame(nullptr)
{
    tracker = new DenseTracking(g_width[0], g_height[0], g_calib[0].cast<double>(), NUM_PYR, {10, 5, 3, 3, 3}, g_bUseColour, g_bUseDepth);
}

void Tracking::trackImage(cv::Mat image, cv::Mat depth, const double timeStamp)
{
    currentFrame = new Frame(image, depth, timeStamp);

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

    if (lastFrame && !lastFrame->mbIsKeyFrame)
        delete lastFrame;

    lastFrame = currentFrame;
}

void Tracking::initialisation()
{
    tracker->SetReferenceImage(currentFrame->mImGray);
    tracker->SetReferenceDepth(currentFrame->mImDepth);

    currentFrame->mTcw = Sophus::SE3d(Eigen::Matrix4d::Identity());
    T_ref2World = currentFrame->mTcw;
    mapping->addKeyFrameCandidate(currentFrame);

    trackingState = OK;
}

bool Tracking::trackLastFrame()
{
    tracker->SetTrackingImage(currentFrame->mImGray);
    tracker->SetTrackingDepth(currentFrame->mImDepth);

    Sophus::SE3d Tpc = tracker->GetTransform();

    currentFrame->mTcw = lastFrame->mTcw * Tpc.inverse();

    if (g_bEnableViewer)
        viewer->setLivePose(currentFrame->mTcw.matrix());

    return true;
}

bool Tracking::relocalisation()
{
    return false;
}

bool Tracking::NeedNewKeyFrame()
{

    Sophus::SE3d DT = T_ref2World.inverse() * currentFrame->mTcw;

    if (DT.log().topRows<3>().norm() > 0.3)
        return true;

    if (DT.log().bottomRows<3>().norm() > 0.3)
        return true;

    return false;
}

void Tracking::MakeNewKeyFrame()
{
    T_ref2World = currentFrame->mTcw;
    currentFrame->mbIsKeyFrame = true;
    mapping->addKeyFrameCandidate(currentFrame);
}

void Tracking::reset()
{
    trackingState = Null;
    lastFrame = NULL;
    currentFrame = NULL;
}

} // namespace SLAM