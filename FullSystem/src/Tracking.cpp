#include "Tracking.h"
#include <future>

namespace SLAM
{

Tracking::Tracking(System *pSystem, Map *pMap)
    : mpSystem(pSystem), mpMap(pMap), mState(SYSTEM_NOT_READY)
{
    int w = g_width[0];
    int h = g_height[0];
    Eigen::Matrix3f calib = g_calib[0];

    mpTracker = new RGBDTracking(w, h, calib.cast<double>(), g_bUseColour, g_bUseDepth);
    mpMapper = new VoxelMapping(w, h, g_calib[0]);
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

void Tracking::GrabImageRGBD(cv::Mat ImgGray, cv::Mat ImgDepth, const double TimeStamp)
{
    mCurrentFrame = Frame(ImgGray, ImgDepth, TimeStamp, g_pORBExtractor);

    Track();
}

void Tracking::Track()
{
    bool bOK = false;
    switch (mState)
    {
    case SYSTEM_NOT_READY:
        InitializeSystem();
        if (mState != OK)
            return;
        break;

    case OK:
        bOK = TrackRGBD();

        if (bOK)
        {
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();
        }
        else
        {
            bOK = Relocalization();
            if (!bOK)
            {
                mState = LOST;
            }
        }
        break;

    case LOST:
        bOK = Relocalization();

        if (bOK)
        {
            mState = OK;
            break;
        }
        else
            return;
    }

    mLastFrame = Frame(mCurrentFrame);
}

void Tracking::InitializeSystem()
{
    mpTracker->SetReferenceImage(mCurrentFrame.mImGray);
    mpTracker->SetReferenceDepth(mCurrentFrame.mImDepth);
    mpMapper->FuseFrame(cv::cuda::GpuMat(mCurrentFrame.mImDepth), mCurrentFrame.mTcw);
    mpLocalMapper->AddKeyFrameCandidate(mCurrentFrame);
    mState = OK;
}

bool Tracking::TrackRGBD()
{
    // Set tracking frames
    mpTracker->SetTrackingImage(mCurrentFrame.mImGray);
    mpTracker->SetTrackingDepth(mCurrentFrame.mImDepth);

    // Calculate the relateive transformation
    Sophus::SE3d DT = mpTracker->GetTransform(mLastFrame.mRelativePose.inverse(), false);

    mCurrentFrame.mRelativePose = DT.inverse();
    mCurrentFrame.mTcw = mReferenceFramePose * DT.inverse();

    mpMapper->FuseFrame(cv::cuda::GpuMat(mCurrentFrame.mImDepth), mCurrentFrame.mTcw);
    g_nTrackedFrame++;

    if (mpViewer)
    {
        mpViewer->setLivePose(mCurrentFrame.mTcw.matrix());
    }

    return true;
}

bool Tracking::Relocalization()
{
    return false;
}

bool Tracking::NeedNewKeyFrame()
{
    Sophus::SE3d DT = mCurrentFrame.mRelativePose;

    if (DT.log().topRows<3>().norm() > 0.15)
        return true;

    if (DT.log().bottomRows<3>().norm() > 0.2)
        return true;

    return false;
}

void Tracking::CreateNewKeyFrame()
{
    // Update the reference pose
    mReferenceFramePose = mCurrentFrame.mTcw;

    // Create a new keyframe
    mpLocalMapper->AddKeyFrameCandidate(mCurrentFrame);

    // Swap the dense tracking buffer
    mpTracker->SwapFrameBuffer();
    mCurrentFrame.mRelativePose = Sophus::SE3d();

    mpMap->AddMapStruct(mpMapper->GetMapStruct());
}

void Tracking::reset()
{
    mState = SYSTEM_NOT_READY;
}

} // namespace SLAM