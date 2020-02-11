#include "Tracking.h"

namespace SLAM
{

Tracking::Tracking(System *pSys, Map *pMap, Viewer *pViewer, Mapping *pMapping)
    : mpFullSystem(pSys), mpMap(pMap), mbOnlyTracking(false), viewer(pViewer),
      mapping(pMapping), mTrackingState(TrackingState::NotInitialized)
{
    mpORBextractor = new ORB_SLAM2::ORBextractor(g_ORBNFeatures, g_ORBScaleFactor, g_ORBNLevels, g_ORBIniThFAST, g_ORBMinThFAST);
    tracker = new DenseTracking(g_width[0], g_height[0], g_calib[0].cast<double>(), NUM_PYR, {10, 5, 3, 3, 3}, g_bUseColour, g_bUseDepth);
}

void Tracking::trackImage(const cv::Mat &imGray, const cv::Mat &imDepth, double TimeStamp)
{
    mCurrentFrame = Frame(imGray, imDepth, TimeStamp, mpORBextractor, mpORBVocabulary);

    bool bOK = false;
    switch (mTrackingState)
    {
    case TrackingState::NotInitialized:
    {
        // Try to initialize the system
        Initialization();

        if (mTrackingState != TrackingState::OK)
            return;
    }

    case TrackingState::OK:
    {
        if (!mbOnlyTracking)
        {
            bOK = TrackLastFrame();
        }
        else
        {
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        if (bOK)
        {
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();
        }
        else
        {
            mTrackingState = TrackingState::Lost;
        }

        break;
    }

    case TrackingState::Lost:
    {
        bool bOK = Relocalization();

        break;
    }
    }

    if (bOK)
        mLastFrame = Frame(mCurrentFrame);
}

void Tracking::Initialization()
{
    mCurrentFrame.ExtractORB();

    if (mCurrentFrame.N > 500)
    {
        // Create the initial keyframe
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap);

        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                Eigen::Vector3d x3D = pKFini->UnprojectKeyPoint(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, pKFini, i);
                pNewMP->AddObservation(pKFini, i);
                pKFini->AddMapPoint(pNewMP, i);
                mpMap->AddMapPoint(pNewMP);
            }
        }

        // Insert KeyFrame into the map
        mpMap->AddKeyFrame(pKFini);

        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mTrackingState = TrackingState::OK;
        tracker->SetReferenceImage(mCurrentFrame.mImGray);
        tracker->SetReferenceDepth(mCurrentFrame.mImDepth);
    }
}

bool Tracking::TrackLastFrame()
{
    tracker->SetTrackingImage(mCurrentFrame.mImGray);
    tracker->SetTrackingDepth(mCurrentFrame.mImDepth);

    Sophus::SE3d Tpc = tracker->GetTransform();

    mCurrentFrame.mTcw = mLastFrame.mTcw * Tpc.inverse();

    // Update the viewer
    if (viewer)
        viewer->setLivePose(mCurrentFrame.mTcw.matrix());

    return true;
}

bool Tracking::Relocalization()
{
}

bool Tracking::NeedNewKeyFrame()
{
    if (mbOnlyTracking)
        return false;

    if (!mpReferenceKF)
        return false;

    Sophus::SE3d DT = mpReferenceKF->mTcw.inverse() * mCurrentFrame.mTcw;

    if (DT.log().topRows<3>().norm() > 0.3)
        return true;

    if (DT.log().bottomRows<3>().norm() > 0.3)
        return true;

    // criteria 1: when observed points falls bellow a threshold
    // if (mObs < 200 || mObsRatio <= 0.4)
    //     return true;

    return false;
}

void Tracking::CreateNewKeyFrame()
{
    mCurrentFrame.ExtractORB();

    if (mCurrentFrame.N > 500)
    {
        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap);
        size_t nObsMPs = 0;
        pKF->mvpObservedMapPoints.clear();

        for (int i = 0; i < mpReferenceKF->mvpMapPoints.size(); ++i)
        {
            MapPoint *pMP = mpReferenceKF->mvpMapPoints[i];
            if (pMP && pKF->IsInFrustum(pMP, 0.5))
            {
                pKF->mvpObservedMapPoints.push_back(pMP);
                nObsMPs++;
            }
        }

        mapping->InsertKeyFrame(pKF);
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;
    }
}

void Tracking::reset()
{
    mTrackingState = TrackingState::NotInitialized;
    mLastFrame.mTcw = Sophus::SE3d();
}

} // namespace SLAM