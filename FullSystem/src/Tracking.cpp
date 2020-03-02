#include "Tracking.h"
#include <future>

namespace SLAM
{

Tracking::Tracking(System *pSystem, Map *pMap)
    : mpSystem(pSystem), mpMap(pMap), mState(SYSTEM_NOT_READY),
      mpCurrentKeyFrame(NULL), mpLastKeyFrame(NULL)
{
    int w = g_width[0];
    int h = g_height[0];
    Eigen::Matrix3f calib = g_calib[0];

    mpTracker = new RGBDTracking(w, h, calib.cast<double>(), g_bUseColour, g_bUseDepth);
    mpMeshEngine = new MeshEngine(20000000);
    mpExtractor = new ORBextractor(g_ORBNFeatures, g_ORBScaleFactor, g_ORBNLevels, g_ORBIniThFAST, g_ORBMinThFAST);
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
    mCurrentFrame = Frame(ImgGray, ImgDepth, TimeStamp, mpExtractor);

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
    KeyFrame *KFinit = new KeyFrame(mCurrentFrame, mpMap);
    if (KFinit->N == 0)
    {
        delete KFinit;
        return;
    }

    mpCurrentMapStruct = new MapStruct(g_calib[0]);
    mpCurrentMapStruct->setMeshEngine(mpMeshEngine);
    mpCurrentMapStruct->create(20000, 10000, 15000, 0.01, 0.05);
    mpCurrentMapStruct->Reset();

    mpCurrentKeyFrame = KFinit;
    mpCurrentKeyFrame->mpVoxelStruct = mpCurrentMapStruct;
    mpCurrentKeyFrame->mbVoxelStructMarginalized = false;

    mpTracker->SetReferenceImage(mCurrentFrame.mImGray);
    mpTracker->SetReferenceDepth(mCurrentFrame.mImDepth);
    mpCurrentMapStruct->Fuse(cv::cuda::GpuMat(mCurrentFrame.mImDepth), mCurrentFrame.mTcw);

    mpLocalMapper->AddKeyFrameCandidate(mpCurrentKeyFrame);
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

    mpCurrentMapStruct->Fuse(cv::cuda::GpuMat(mCurrentFrame.mImDepth), mCurrentFrame.mRelativePose);
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

    if (DT.log().topRows<3>().norm() > 0.1)
        return true;

    if (DT.log().bottomRows<3>().norm() > 0.15)
        return true;

    return false;
}

void Tracking::CreateNewKeyFrame()
{
    // Update the reference pose
    mReferenceFramePose = mCurrentFrame.mTcw;
    mpMap->AddMapStruct(mpCurrentMapStruct);

    // Create a new MapStruct for the new Keyframe
    mpCurrentMapStruct = new MapStruct(g_calib[0]);
    mpCurrentMapStruct->setMeshEngine(mpMeshEngine);
    mpCurrentMapStruct->create(20000, 15000, 15000, 0.008, 0.03);
    mpCurrentMapStruct->Reset();
    mpCurrentMapStruct->mTcw = mCurrentFrame.mTcw;

    // Create a new keyframe
    KeyFrame *pNewKF = new KeyFrame(mCurrentFrame, mpMap);
    pNewKF->mpVoxelStruct = mpCurrentMapStruct;
    pNewKF->mbVoxelStructMarginalized = false;
    mpCurrentMapStruct->mTcw = pNewKF->mTcw;
    mpLocalMapper->AddKeyFrameCandidate(pNewKF);

    // Swap the dense tracking buffer
    mpTracker->SwapFrameBuffer();
    mCurrentFrame.mRelativePose = Sophus::SE3d();
}

void Tracking::reset()
{
    mState = SYSTEM_NOT_READY;
}

} // namespace SLAM