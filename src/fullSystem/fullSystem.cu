#include "fullSystem/fullSystem.h"
#include "denseTracker/cudaImageProc.h"

FullSystem::FullSystem(const char *configFile)
    : viewerEnabled(false)
{
}

FullSystem::FullSystem(
    int w, int h,
    Mat33d K,
    int numLvl,
    bool enableViewer)
    : currentState(-1),
      viewerEnabled(enableViewer)
{
    globalMapper = std::make_shared<GlobalMapper>(K, 5);
    localMapper = std::make_shared<DenseMapping>(w, h, K);
    coarseTracker = std::make_shared<DenseTracker>(w, h, K, numLvl);

    lastTrackedPose = SE3(Mat44d::Identity());
    accumulateTransform = SE3(Mat44d::Identity());

    bufferVec4wxh.create(h, w, CV_32FC4);
    bufferFloatwxh.create(h, w, CV_32FC1);

    optThread = std::thread(&GlobalMapper::optimizationLoop, globalMapper.get());
    loopThread = std::thread(&GlobalMapper::globalConsistencyLoop, globalMapper.get());
}

FullSystem::~FullSystem()
{
    globalMapper->setShouldQuit();
    optThread.join();
    loopThread.join();
}

void FullSystem::processFrame(Mat rawImage, Mat rawDepth)
{
    Mat rawImageFloat, rawIntensity;
    rawImage.convertTo(rawImageFloat, CV_32FC3);
    cv::cvtColor(rawImageFloat, rawIntensity, cv::COLOR_RGB2GRAY);
    currentFrame = std::make_shared<Frame>(rawImage, rawDepth, rawIntensity);

    switch (currentState)
    {
    case -1:
    {
        if (viewerEnabled && viewer)
            viewer->setCurrentState(-1);

        coarseTracker->setReferenceFrame(currentFrame);
        createNewKF();
        fuseCurrentFrame();
        currentState = 0;

        if (viewerEnabled && viewer)
            viewer->setCurrentState(0);

        break;
    }
    case 0:
    {
        auto rval = trackCurrentFrame();
        if (rval)
        {
            fuseCurrentFrame();
            raytraceCurrentFrame();

            if (needNewKF())
                createNewKF();
            else
                globalMapper->addFrameHistory(currentFrame);

            if (viewerEnabled && viewer)
                viewer->addTrackingResult(currentFrame->getPoseInLocalMap());
        }
        else
        {
            if (viewerEnabled && viewer)
                viewer->setCurrentState(1);

            currentState = 1;
        }

        break;
    }
    case 1:

        size_t numAttempted = 0;
        printf("tracking loast, attempt to resuming...\n");
        while (numAttempted <= maxNumRelocAttempt)
        {
            if (tryRelocalizeCurrentFrame(numAttempted > 0))
            {
                if (viewerEnabled && viewer)
                    viewer->setCurrentState(0);

                break;
            }
        }

        break;
    }
}

bool FullSystem::trackCurrentFrame()
{
    coarseTracker->setTrackingFrame(currentFrame);
    SE3 tRes = coarseTracker->getIncrementalTransform();
    // accumulated local transform
    accumulateTransform = accumulateTransform * tRes.inverse();
    currentFrame->setTrackingResult(accumulateTransform);
    currentFrame->setReferenceKF(referenceFrame);

    return true;
}

void FullSystem::fuseCurrentFrame()
{
    auto currDepth = coarseTracker->getReferenceDepth();
    localMapper->fuseFrame(currDepth, currentFrame->getPoseInLocalMap());
}

void FullSystem::raytraceCurrentFrame()
{
    localMapper->raytrace(bufferVec4wxh, currentFrame->getPoseInLocalMap());
    coarseTracker->setReferenceInvDepth(bufferVec4wxh);
}

bool FullSystem::tryRelocalizeCurrentFrame(bool updatePoints)
{
    return true;
}

bool FullSystem::needNewKF()
{
    auto dt = currentFrame->getTrackingResult();
    Vec3d t = dt.translation();
    if (t.norm() >= 0.1)
        return true;

    Vec3d r = dt.log().tail<3>();
    if (r.norm() >= 0.1)
        return true;

    return false;
}

void FullSystem::createNewKF()
{
    referenceFrame = currentFrame;

    /* 
        Flag for keyframe
    */
    referenceFrame->flagKeyFrame();
    lastTrackedPose = lastTrackedPose * accumulateTransform;
    referenceFrame->setRawKeyframePose(lastTrackedPose);
    globalMapper->addReferenceFrame(referenceFrame);

    rawKeyFramePoseHistory.push_back(lastTrackedPose);
    accumulateTransform = SE3();
}

void FullSystem::resetSystem()
{
    currentState = -1;
    localMapper->reset();
    globalMapper->reset();
    rawFramePoseHistory.clear();
    rawKeyFramePoseHistory.clear();

    lastTrackedPose = SE3(Mat44d::Identity());
    accumulateTransform = SE3(Mat44d::Identity());
}

std::vector<SE3> FullSystem::getRawFramePoseHistory() const
{
    return rawFramePoseHistory;
}

std::vector<SE3> FullSystem::getRawKeyFramePoseHistory() const
{
    return rawKeyFramePoseHistory;
}

size_t FullSystem::getMesh(float *vbuffer, float *nbuffer, size_t bufferSize)
{
    return localMapper->fetch_mesh_with_normal(vbuffer, nbuffer);
}

std::vector<SE3> FullSystem::getKeyFramePoseHistory()
{
    return globalMapper->getKeyFrameHistory();
}

std::vector<SE3> FullSystem::getFramePoseHistory()
{
    return globalMapper->getFrameHistory();
}

std::vector<Vec3f> FullSystem::getActiveKeyPoints()
{
    return globalMapper->getActivePoints();
}

std::vector<Vec3f> FullSystem::getStableKeyPoints()
{
    return globalMapper->getStablePoints();
}

void FullSystem::setMapViewerPtr(MapViewer *viewer)
{
    this->viewer = viewer;
}