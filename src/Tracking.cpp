#include "Tracking.h"

namespace SLAM
{

Tracking::Tracking(System *system, Map *map, Viewer *viewer, Mapping *mapping)
    : mpFullSystem(system), mpMap(map), viewer(viewer), mapping(mapping),
      mbOnlyTracking(false), mTrackingState(TrackingState::NotInitialized),
      currentFrame(nullptr), lastFrame(nullptr)
{
    // mpORBextractor = new ORB_SLAM2::ORBextractor(g_ORBNFeatures, g_ORBScaleFactor, g_ORBNLevels, g_ORBIniThFAST, g_ORBMinThFAST);
    tracker = new DenseTracking(g_width[0], g_height[0], g_calib[0].cast<double>(), NUM_PYR, {10, 5, 3, 3, 3}, g_bUseColour, g_bUseDepth);
}

void Tracking::trackImage(cv::Mat image, cv::Mat depth, const double timeStamp)
{
    currentFrame = new Frame(image, depth, timeStamp);

    bool bOK = false;
    switch (mTrackingState)
    {
    case TrackingState::NotInitialized:
    {
        initialisation();
        break;
    }

    case TrackingState::OK:
    {
        bool bOK = trackLastFrame();

        if (bOK)
        {
            if (needNewKeyFrame())
                addKeyFrameCandidate();
        }
        else
        {
            mTrackingState = TrackingState::Lost;
        }

        break;
    }

    case TrackingState::Lost:
    {
        bool bOK = relocalisation();

        if (bOK)
        {
            mTrackingState = TrackingState::OK;
            break;
        }
        else
            return;
    }
    }

    lastFrame = currentFrame;
}

void Tracking::initialisation()
{
    tracker->SetReferenceImage(currentFrame->mImGray);
    tracker->SetReferenceDepth(currentFrame->mImDepth);

    currentFrame->mTcw = Sophus::SE3d(Eigen::Matrix4d::Identity());
    T_ref2World = currentFrame->mTcw;
    mapping->addKeyFrameCandidate(currentFrame);

    mTrackingState = TrackingState::OK;
}

bool Tracking::trackLastFrame()
{
    tracker->SetTrackingImage(currentFrame->mImGray);
    tracker->SetTrackingDepth(currentFrame->mImDepth);

    Sophus::SE3d Tpc = tracker->GetTransform();

    currentFrame->mTcw = lastFrame->mTcw * Tpc.inverse();

    // Update the viewer
    if (viewer)
        viewer->setLivePose(currentFrame->mTcw.matrix());

    return true;
}

bool Tracking::relocalisation()
{
    return false;
}

bool Tracking::needNewKeyFrame()
{
    if (mbOnlyTracking)
        return false;

    if (!mpReferenceKF)
        return false;

    Sophus::SE3d DT = T_ref2World.inverse() * currentFrame->mTcw;

    if (DT.log().topRows<3>().norm() > 0.3)
        return true;

    if (DT.log().bottomRows<3>().norm() > 0.3)
        return true;

    // criteria 1: when observed points falls bellow a threshold
    // if (mObs < 200 || mObsRatio <= 0.4)
    //     return true;

    return false;
}

void Tracking::addKeyFrameCandidate()
{
    // mCurrentFrame.ExtractORB();

    // if (mCurrentFrame.N > 500)
    // {
    //     KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap);
    //     size_t nObsMPs = 0;
    //     pKF->mvpObservedMapPoints.clear();

    //     for (int i = 0; i < mpReferenceKF->mvpMapPoints.size(); ++i)
    //     {
    //         MapPoint *pMP = mpReferenceKF->mvpMapPoints[i];
    //         if (pMP && pKF->IsInFrustum(pMP, 0.5))
    //         {
    //             pKF->mvpObservedMapPoints.push_back(pMP);
    //             nObsMPs++;
    //         }
    //     }

    //     mapping->InsertKeyFrame(pKF);
    //     mpReferenceKF = pKF;
    //     mCurrentFrame.mpReferenceKF = pKF;
    // }
    // mapping->InsertKeyFrame(pKF);
    // mapping->addKeyFrameCandidate(currentFrame);
    T_ref2World = currentFrame->mTcw;
    mapping->addKeyFrameCandidate(currentFrame);
}

void Tracking::reset()
{
    mTrackingState = TrackingState::NotInitialized;
    lastFrame = NULL;
    currentFrame = NULL;
}

} // namespace SLAM