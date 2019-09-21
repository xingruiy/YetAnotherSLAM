#include "fullSystem/fullSystem.h"
#include "denseTracker/cudaImageProc.h"

FullSystem::FullSystem(const char *configFile)
{
}

FullSystem::FullSystem(int w, int h, Mat33d K, int numLvl, bool optimize)
    : currentState(-1)
{
    localMapper = std::make_shared<DenseMapping>(w, h, K);
    globalMapper = std::make_shared<GlobalMapper>(K, 7);
    coarseTracker = std::make_shared<DenseTracker>(w, h, K, numLvl);

    lastTrackedPose = SE3(Mat44d::Identity());
    lastReferencePose = SE3(Mat44d::Identity());
    accumulateTransform = SE3(Mat44d::Identity());

    bufferVec4wxh.create(h, w, CV_32FC4);
    bufferFloatwxh.create(h, w, CV_32FC1);

    optThread = std::thread(&GlobalMapper::optimizationLoop, globalMapper.get());
}

FullSystem::~FullSystem()
{
    globalMapper->setShouldQuit();
    optThread.join();
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
        coarseTracker->setReferenceFrame(currentFrame);
        createNewKF();
        fuseCurrentFrame(lastTrackedPose);
        currentState = 0;
        break;
    }
    case 0:
    {
        auto rval = trackCurrentFrame();
        if (rval)
        {
            fuseCurrentFrame(lastTrackedPose);
            updateLocalMapObservation(lastTrackedPose);

            if (needNewKF())
                createNewKF();
            else
                globalMapper->addFrameHistory(currentFrame);
            rawFramePoseHistory.push_back(currentFrame->getPoseInLocalMap());
        }
        else
        {
            currentState = 1;
        }

        break;
    }
    case 1:
        printf("tracking loast, attempt to resuming...\n");
        break;
    }
}

bool FullSystem::trackCurrentFrame()
{
    coarseTracker->setTrackingFrame(currentFrame);
    // SE3 relativePose = coarseTracker->getIncrementalTransform();
    // lastTrackedPose = lastTrackedPose * relativePose.inverse();
    // currentFrame->setPose(lastTrackedPose);
    // currentFrame->relativePose = relativePose;
    // currentFrame->referenceKF = referenceFrame;

    SE3 tRes = coarseTracker->getIncrementalTransform();
    // accumulated local transform
    accumulateTransform = accumulateTransform * tRes.inverse();
    currentFrame->setTrackingResult(accumulateTransform);
    currentFrame->setReferenceKF(referenceFrame);

    // Keep track of the camera in local map
    // lastTrackedPose = lastTrackedPose * tRes.inverse();
    // currentFrame->setPose(lastTrackedPose);

    return true;
}

void FullSystem::fuseCurrentFrame(const SE3 &T)
{
    // SE3 transformedPose = SE3();
    // if (currentFrame->isKeyframe())
    //     transformedPose = currentFrame->accumulatedPose;
    // else
    // {
    //     auto relativePose = currentFrame->relativePose;
    //     auto referencePose = currentFrame->referenceKF->accumulatedPose;
    //     transformedPose = referencePose * relativePose.inverse();
    // }

    auto currDepth = coarseTracker->getReferenceDepth();
    localMapper->fuseFrame(currDepth, currentFrame->getPoseInLocalMap());
    // localMapper->fuseFrame(currDepth, transformedPose);
}

void FullSystem::updateLocalMapObservation(const SE3 &T)
{
    localMapper->raytrace(bufferVec4wxh, currentFrame->getPoseInLocalMap());
    coarseTracker->setReferenceInvDepth(bufferVec4wxh);
}

bool FullSystem::needNewKF()
{
    auto dt = currentFrame->getTrackingResult();
    // SE3 dt = referencePose * lastTrackedPose.inverse();
    Vec3d t = dt.translation();
    if (t.norm() >= 0.1)
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
    // lastReferencePose = lastTrackedPose;
}

void FullSystem::resetSystem()
{
    currentState = -1;
    localMapper->reset();
    globalMapper->reset();
    rawFramePoseHistory.clear();
    rawKeyFramePoseHistory.clear();

    lastTrackedPose = SE3(Mat44d::Identity());
    lastReferencePose = SE3(Mat44d::Identity());
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

std::vector<Vec3f> FullSystem::getActiveKeyPoints()
{
    return globalMapper->getActivePoints();
}

std::vector<Vec3f> FullSystem::getStableKeyPoints()
{
    return globalMapper->getStablePoints();
}