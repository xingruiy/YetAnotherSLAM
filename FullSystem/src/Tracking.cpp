#include "Tracking.h"
#include <future>

namespace SLAM
{

Tracking::Tracking(System *system, Map *map, Viewer *mpViewer, LocalMapping *mpLocalMapping)
    : mpSystem(system), mpMap(map), mpViewer(mpViewer), mpLocalMapping(mpLocalMapping), trackingState(Null)
{
    int w = g_width[0];
    int h = g_height[0];
    Eigen::Matrix3f calib = g_calib[0];

    mpTracker = new DenseTracking(w, h, calib.cast<double>(), NUM_PYR, {10, 5, 3, 3, 3}, g_bUseColour, g_bUseDepth);
    mpMapper = new DenseMapping(w, h, g_calib[0]);
    mCurrentMapPrediction.create(h, w, CV_32FC4);
}

void Tracking::trackImage(cv::Mat ImgGray, cv::Mat ImgDepth, const double TimeStamp)
{
    NextFrame = Frame(ImgGray, ImgDepth, TimeStamp, g_pORBExtractor);
    std::future<void> result = std::async(std::launch::async, &Frame::ExtractORBFeatures, &NextFrame);

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
        auto t1 = std::chrono::high_resolution_clock::now();
        bool bOK = trackLastFrame();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "tracking: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

        t1 = std::chrono::high_resolution_clock::now();
        result.get();
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "waited: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

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
    mpMapper->fuseFrame(cv::cuda::GpuMat(NextFrame.mImDepth), NextFrame.mTcw);
    mpLocalMapping->AddKeyFrameCandidate(NextFrame);
    trackingState = OK;
}

bool Tracking::trackLastFrame()
{
    mpTracker->SetTrackingImage(NextFrame.mImGray);
    mpTracker->SetTrackingDepth(NextFrame.mImDepth);
    mpMapper->raytrace(mCurrentMapPrediction, lastFrame.mTcw);

    Sophus::SE3d Tpc = mpTracker->GetTransform(lastFrame.T_frame2Ref.inverse(), false);

    NextFrame.mTcw = T_ref2World * Tpc.inverse();
    NextFrame.T_frame2Ref = Tpc.inverse();

    if (g_bEnableViewer)
        mpViewer->setLivePose(NextFrame.mTcw.matrix());

    mpMapper->fuseFrame(cv::cuda::GpuMat(NextFrame.mImDepth), NextFrame.mTcw);
    g_nTrackedFrame++;
    return true;
}

bool Tracking::Relocalisation()
{
    return false;
}

bool Tracking::NeedNewKeyFrame()
{
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