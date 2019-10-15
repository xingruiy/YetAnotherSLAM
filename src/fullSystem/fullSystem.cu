#include "fullSystem/fullSystem.h"
#include "denseTracker/cudaImageProc.h"

FullSystem::FullSystem(
    int w, int h,
    Mat33d K,
    int numLvl,
    bool enableViewer)
    : state(SystemState::NotInitialized),
      lastState(SystemState::NotInitialized),
      viewerEnabled(enableViewer),
      mappingEnabled(true),
      imageWidth(w),
      imageHeight(h),
      camIntrinsics(K),
      numProcessedFrames(0)
{
    map = std::make_shared<Map>();
    localOptimizer = std::make_shared<LocalOptimizer>(K, 3, map);
    localMapper = std::make_shared<DenseMapping>(w, h, K);
    coarseTracker = std::make_shared<DenseTracker>(w, h, K, numLvl);

    lastTrackedPose = SE3(Mat44d::Identity());
    accumulateTransform = SE3(Mat44d::Identity());

    bufferVec4wxh.create(h, w, CV_32FC4);
    bufferFloatwxh.create(h, w, CV_32FC1);

    localOptThread = std::thread(&LocalOptimizer::loop, localOptimizer.get());
}

FullSystem::~FullSystem()
{
    localOptimizer->setShouldQuit();
    printf("wating other threads to finish...\n");
    localOptThread.join();
    printf("all threads finished!\n");
}

void FullSystem::processFrame(Mat rawImage, Mat rawDepth)
{
    rawImage.convertTo(cbufferFloatVec3wxh, CV_32FC3);
    cv::cvtColor(cbufferFloatVec3wxh, cbufferFloatwxh, cv::COLOR_RGB2GRAY);

    currentFrame = std::make_shared<Frame>(
        imageWidth,
        imageHeight,
        camIntrinsics,
        rawImage,
        rawDepth,
        cbufferFloatwxh);

    switch (state)
    {
    case SystemState::NotInitialized:
    {
        if (viewerEnabled && viewer)
            viewer->setCurrentState(-1);

        coarseTracker->setReferenceFrame(currentFrame);
        createNewKF();
        fuseCurrentFrame();
        state = SystemState::OK;

        if (viewerEnabled && viewer)
            viewer->setCurrentState(0);

        break;
    }
    case SystemState::OK:
    {
        auto rval = trackCurrentFrame();
        if (rval)
        {
            fuseCurrentFrame();
            raytraceCurrentFrame();

            if (needNewKF())
            {
                createNewKF();
            }
            else
            {
                map->addFramePose(currentFrame->getTrackingResult(), currentKeyframe);
            }

            if (viewerEnabled && viewer)
                viewer->addTrackingResult(currentFrame->getPoseInLocalMap());
        }
        else
        {
            if (viewerEnabled && viewer)
                viewer->setCurrentState(1);

            state = SystemState::Lost;
        }

        break;
    }
    case SystemState::Lost:

        size_t numAttempted = 0;
        printf("tracking loast, attempt to resuming...\n");
        while (numAttempted <= maxNumRelocAttempt)
        {
            if (tryRelocalizeCurrentFrame())
            {
                if (viewerEnabled && viewer)
                    viewer->setCurrentState(0);

                break;
            }
        }

        break;
    }

    lastState = state;
    if (state == SystemState::OK)
        numProcessedFrames++;
}

bool FullSystem::trackCurrentFrame()
{
    coarseTracker->setTrackingFrame(currentFrame);
    SE3 tRes = coarseTracker->getIncrementalTransform();
    // accumulated local transform
    accumulateTransform = accumulateTransform * tRes.inverse();
    currentFrame->setTrackingResult(accumulateTransform);
    currentFrame->setReferenceKF(currentKeyframe);

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

bool FullSystem::tryRelocalizeCurrentFrame()
{
    auto matcher = std::make_shared<FeatureMatcher>(PointType::ORB, DescType::ORB);
    currentFrame->detectKeyPoints(matcher);
    Mat descAll;
    const auto desc = map->getPointDescriptorsAll();
    std::vector<std::vector<cv::DMatch>> rawMatches;
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher2 = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    matcher2->knnMatch(currentFrame->pointDesc, desc, rawMatches, 2);

    for (auto knn : rawMatches)
    {
        if (knn[0].distance / knn[1].distance < 0.8)
            matches.push_back(knn[0]);
    }

    const auto &pts = map->getMapPointsAll();

    std::vector<Vec3f> matchedPoints;
    for (auto m : matches)
    {
        if (pts[m.trainIdx] && !pts[m.trainIdx]->isBad())
            matchedPoints.push_back(pts[m.trainIdx]->getPosWorld().cast<float>());
    }

    if (viewerEnabled && viewer)
        viewer->setMatchedPoints(matchedPoints);

    std::cout << matchedPoints.size() << std::endl;

    return true;
}

bool FullSystem::needNewKF()
{
    auto dt = currentFrame->getTrackingResult();
    Vec3d t = dt.translation();
    if (t.norm() >= 0.3)
        return true;

    Vec3d r = dt.log().tail<3>();
    if (r.norm() >= 0.3)
        return true;

    return false;
}

void FullSystem::createNewKF()
{
    currentKeyframe = currentFrame;

    currentKeyframe->flagKeyFrame();
    lastTrackedPose = lastTrackedPose * accumulateTransform;
    currentKeyframe->setRawKeyframePose(lastTrackedPose);

    if (mappingEnabled)
    {
        map->addUnprocessedKeyframe(currentKeyframe);
        // map->setCurrentKeyframe(currentKeyframe);
        map->addKeyframePoseRaw(lastTrackedPose);
        map->addFramePose(SE3(), currentKeyframe);
    }

    if (viewerEnabled && viewer)
        viewer->addRawKeyFramePose(lastTrackedPose);

    accumulateTransform = SE3();
}

void FullSystem::resetSystem()
{
    map->clear();
    viewer->resetViewer();
    localMapper->reset();
    localOptimizer->reset();
    lastTrackedPose = SE3(Mat44d::Identity());
    accumulateTransform = SE3(Mat44d::Identity());
    state = SystemState::NotInitialized;
    lastState = SystemState::NotInitialized;
}

size_t FullSystem::getMesh(float *vbuffer, float *nbuffer, size_t bufferSize)
{
    return localMapper->fetchMeshWithNormal(vbuffer, nbuffer);
}

std::vector<SE3> FullSystem::getKeyFramePoseHistory()
{
    return map->getKeyframePoseOptimized();
}

std::vector<SE3> FullSystem::getFramePoseHistory()
{
    return map->getFramePoseOptimized();
}

std::vector<Vec3f> FullSystem::getMapPointPosAll()
{
    return map->getMapPointVec3All();
}

void FullSystem::setMapViewerPtr(MapViewer *viewer)
{
    this->viewer = viewer;
}

void FullSystem::setMappingEnable(const bool enable)
{
    mappingEnabled = enable;
}

void FullSystem::setSystemStateToLost()
{
    state = SystemState::Lost;
}