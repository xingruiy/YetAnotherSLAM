#include "fullSystem/fullSystem.h"
#include "localizer/localizer.h"
#include "denseTracker/cudaImageProc.h"

FullSystem::FullSystem(
    int w, int h,
    Mat33d K,
    int numLvl,
    bool enableViewer)
    : state(SystemState::NotInitialized),
      lastState(SystemState::NotInitialized),
      viewerEnabled(enableViewer),
      numTimesRun(0),
      testKFId(0),
      lastTestedKFId(0),
      mappingEnabled(true),
      imageWidth(w),
      imageHeight(h),
      camIntrinsics(K),
      numProcessedFrames(0),
      useGraphMatching(false),
      shouldCalculateNormal(false)
{
    map = std::make_shared<Map>();
    localOptimizer = std::make_shared<LocalOptimizer>(K, 3, map);
    localMapper = std::make_shared<DenseMapping>(w, h, K);
    coarseTracker = std::make_shared<DenseTracker>(w, h, K, numLvl);

    lastTrackedPose = SE3(Mat44d::Identity());
    accumulateTransform = SE3(Mat44d::Identity());

    gpuBufferVec4FloatWxH.create(h, w, CV_32FC4);
    gpuBufferFloatWxH.create(h, w, CV_32FC1);

    localOptThread = std::thread(&LocalOptimizer::loop, localOptimizer.get());
}

FullSystem::~FullSystem()
{
    localOptimizer->setShouldQuit();
    printf("wating other threads to finish...\n");
    localOptThread.join();
    printf("all threads finished!\n");
}

void FullSystem::setCurrentNormal(GMat nmap)
{
    nmap.download(cpuBufferVec4FloatWxH);
}

void FullSystem::processFrame(Mat rawImage, Mat rawDepth)
{
    rawImage.convertTo(cpuBufferVec3FloatWxH, CV_32FC3);
    cv::cvtColor(cpuBufferVec3FloatWxH, cpuBufferFloatWxH, cv::COLOR_RGB2GRAY);

    if (lastState != SystemState::Lost)
        currentFrame = std::make_shared<Frame>(
            imageWidth,
            imageHeight,
            camIntrinsics,
            rawImage,
            rawDepth,
            cpuBufferFloatWxH);

    switch (state)
    {
    case SystemState::NotInitialized:
    {
        if (viewerEnabled && viewer)
            viewer->setCurrentState(-1);

        coarseTracker->setReferenceFrame(currentFrame);
        createNewKF();
        fuseCurrentFrame();
        raytraceCurrentFrame();

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
    {
        printf("tracking loast, attempt to resuming...\n");
        if (tryRelocalizeCurrentFrame())
        {
            if (viewerEnabled && viewer)
                viewer->setCurrentState(0);
        }
        break;
    }

    case SystemState::Test:
    {
        if (lastState == SystemState::Test && testKFId == lastTestedKFId)
            break;

        auto KFs = map->getKeyframesAll();
        auto &kf = KFs[testKFId];
        std::cout << "testing kf: " << kf->getId() << std::endl;
        tryRelocalizeKeyframe(KFs[testKFId]);
        lastTestedKFId = testKFId;
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
    localMapper->raytrace(gpuBufferVec4FloatWxH, currentFrame->getPoseInLocalMap());
    coarseTracker->setReferenceInvDepth(gpuBufferVec4FloatWxH);
    computeNormal(gpuBufferVec4FloatWxH, gpuBufferVec4FloatWxH2);
    currentFrame->setNormalMap(Mat(gpuBufferVec4FloatWxH2));
}

bool FullSystem::tryRelocalizeKeyframe(std::shared_ptr<Frame> kf)
{
    Mat descriptor;
    std::vector<float> keyPointDepth;
    std::vector<Vec3f> keyPointNormal;
    std::vector<cv::KeyPoint> cvKeyPoints;

    // detect features for the current frame
    auto matcher = std::make_shared<FeatureMatcher>(PointType::ORB, DescType::ORB);
    matcher->detectAndCompute(kf->getImage(), cvKeyPoints, descriptor);
    matcher->computePointDepth(kf->getOGDepth(), cvKeyPoints, keyPointDepth);

    // draw detected keypoints
    if (viewerEnabled && viewer)
    {
        cv::drawKeypoints(kf->getImage(), cvKeyPoints, cpuBufferVec3ByteWxH, cv::Scalar(255, 0, 0));
        viewer->setKeyPointImage(cpuBufferVec3ByteWxH);
    }

    auto numFeatures = kf->getNumPointsDetected();
    if (numFeatures < 10)
    {
        printf("too few points detected(%lu), relocalization failed...\n", numFeatures);
        return false;
    }

    std::vector<bool> valid(numFeatures);
    std::vector<Vec3d> keyPoint(numFeatures);
    std::fill(valid.begin(), valid.end(), false);
    for (int n = 0; n < numFeatures; ++n)
    {
        auto &z = kf->keyPointDepth[n];
        if (z)
        {
            valid[n] = true;
            auto &kp = cvKeyPoints[n].pt;
            keyPoint[n] = camIntrinsics.inverse() * Vec3d(kp.x, kp.y, 1.0) * z;
        }
    }

    Localizer relocalizer;
    std::vector<SE3> hypothesesList;
    std::vector<std::vector<bool>> filter;
    std::vector<std::vector<cv::DMatch>> matches;
    if (!relocalizer.getRelocHypotheses(
            map,
            keyPoint,
            keyPointNormal,
            descriptor,
            valid,
            hypothesesList,
            matches,
            filter,
            useGraphMatching))
        return false;

    if (hypothesesList.size() == 0)
    {
        printf("too few hypotheses(%lu), relocalization failed...\n", hypothesesList.size());
        return false;
    }

    cv::Mat test;
    // display matched points in the image
    if (matches.size() != 0 &&
        filter.size() != 0 &&
        viewerEnabled && viewer)
    {
        std::vector<cv::KeyPoint> ptMatched;
        std::vector<cv::KeyPoint> ptMatchedDst;
        std::vector<cv::DMatch> ptMatch;
        std::vector<cv::Point> matchingLines;
        auto mapPoints = map->getMapPointsAll();
        const auto &mlist = matches[0];
        const auto &outlier = filter[0];
        const auto &T = hypothesesList[0];
        const auto fx = camIntrinsics(0, 0);
        const auto fy = camIntrinsics(1, 1);
        const auto cx = camIntrinsics(0, 2);
        const auto cy = camIntrinsics(1, 2);
        int counter = 0;
        for (auto i = 0; i < mlist.size(); ++i)
        {
            if (!outlier[i])
            {
                const auto &m = mlist[i];
                const auto &pos = kf->getPoseInGlobalMap().inverse() * mapPoints[m.trainIdx]->getPosWorld();
                const int x = fx * pos(0) / pos(2) + cx;
                const int y = fy * pos(1) / pos(2) + cy;
                if (x >= 0 && y >= 0 && x < 640 && y < 480)
                {
                    ptMatched.push_back(cvKeyPoints[m.queryIdx]);
                    ptMatchedDst.push_back(cv::KeyPoint(x, y, 1));
                    ptMatch.push_back(cv::DMatch(counter, counter, 0));
                    matchingLines.push_back(cv::Point2f(x, y));
                    matchingLines.push_back(cvKeyPoints[m.queryIdx].pt);
                    counter++;
                }
            }
        }

        std::vector<cv::KeyPoint> vcMapPoints;
        for (auto &mp : mapPoints)
        {
            if (mp)
            {
                const auto &pos = kf->getPoseInGlobalMap().inverse() * mp->getPosWorld();
                const int x = fx * pos(0) / pos(2) + cx;
                const int y = fy * pos(1) / pos(2) + cy;
                if (x >= 0 && y >= 0 && x < 640 && y < 480)
                {
                    vcMapPoints.push_back(cv::KeyPoint(x, y, 1));
                }
            }
        }

        // display matched points in a image;
        cv::drawKeypoints(kf->getImage(), vcMapPoints, cpuBufferVec3ByteWxH, cv::Scalar(0, 0, 255));
        cv::drawKeypoints(cpuBufferVec3ByteWxH, ptMatched, cpuBufferVec3ByteWxH, cv::Scalar(0, 255, 0));
        cv::drawKeypoints(cpuBufferVec3ByteWxH, ptMatchedDst, test, cv::Scalar(255, 0, 0));
        for (int i = 0; i < counter - 1; ++i)
        {
            cv::line(test, matchingLines[2 * i], matchingLines[2 * i + 1], cv::Scalar(0, 0, 255));
        }
        // cv::drawMatches(kf->getImage(), ptMatched, kf->getImage(), ptMatchedDst, ptMatch, test);
        viewer->setMatchedPointImage(test);
    }

    // display pose proposals
    if (viewerEnabled && viewer)
        viewer->setRelocalizationHypotheses(hypothesesList);

    // cv::imshow("img", test);
    // cv::waitKey(1);

    return true;
}

bool FullSystem::tryRelocalizeCurrentFrame()
{
    Mat descriptor;
    std::vector<bool> valid;
    std::vector<float> keyPointDepth;
    std::vector<Vec3f> keyPointNormal;
    std::vector<cv::KeyPoint> cvKeyPoint;

    // detect features for the current frame
    auto matcher = std::make_shared<FeatureMatcher>(PointType::ORB, DescType::ORB);
    matcher->detectAndCompute(currentFrame->getImage(), cvKeyPoint, descriptor);
    matcher->computePointDepth(currentFrame->getDepth(), cvKeyPoint, keyPointDepth);

    // draw detected keypoints
    if (viewerEnabled && viewer)
    {
        cv::drawKeypoints(currentFrame->getImage(), cvKeyPoint, cpuBufferVec3ByteWxH, cv::Scalar(255, 0, 0));
        viewer->setKeyPointImage(cpuBufferVec3ByteWxH);
    }

    // calculate normal if needed
    if (shouldCalculateNormal)
        matcher->computePointNormal(cpuBufferVec4FloatWxH, cvKeyPoint, keyPointNormal);

    auto numFeatures = keyPointDepth.size();
    std::vector<Vec3d> keyPoint(numFeatures);

    if (numFeatures < 10)
    {
        printf("too few points detected(%lu), relocalization failed...\n", numFeatures);
        return false;
    }

    valid.resize(numFeatures);
    std::fill(valid.begin(), valid.end(), true);
    for (int n = 0; n < numFeatures; ++n)
    {
        auto &z = keyPointDepth[n];
        if (z > FLT_EPSILON)
        {
            auto &kp = cvKeyPoint[n].pt;
            keyPoint[n] = camIntrinsics.inverse() * Vec3d(kp.x, kp.y, 1.0) * z;
        }
        else
        {
            valid[n] = false;
        }
    }

    Localizer relocalizer;
    std::vector<SE3> hypothesesList;
    std::vector<std::vector<bool>> filter;
    std::vector<std::vector<cv::DMatch>> matches;
    if (!relocalizer.getRelocHypotheses(
            map,
            keyPoint,
            keyPointNormal,
            descriptor,
            valid,
            hypothesesList,
            matches,
            filter,
            useGraphMatching))
        return false;

    if (hypothesesList.size() == 0)
    {
        printf("too few hypotheses(%lu), relocalization failed...\n", hypothesesList.size());
        return false;
    }

    // display matched points in the image
    if (matches.size() != 0 &&
        filter.size() != 0 &&
        viewerEnabled && viewer)
    {
        std::vector<cv::KeyPoint> ptMatched;
        // std::vector<Vec3f> ptMatched3d;
        // std::vector<Vec3f> ptMatchedDst3d;
        // std::vector<Vec3f> lines;
        auto mapPoints = map->getMapPointsAll();
        const auto &mlist = matches[0];
        const auto &outlier = filter[0];
        const auto &T = hypothesesList[0];
        for (auto i = 0; i < mlist.size(); ++i)
        {
            if (!outlier[i])
            {
                const auto &m = mlist[i];
                ptMatched.push_back(cvKeyPoint[m.queryIdx]);
                // ptMatchedDst3d.push_back(mapPoints[m.trainIdx]->getPosWorld().cast<float>());
            }
        }

        // display matched points in a image;
        cv::drawKeypoints(currentFrame->getImage(), ptMatched, cpuBufferVec3ByteWxH, cv::Scalar(0, 255, 0));
        viewer->setMatchedPointImage(cpuBufferVec3ByteWxH);

        auto cx = camIntrinsics(0, 2);
        auto cy = camIntrinsics(1, 2);
        for (int i = 0; i < ptMatched.size(); ++i)
        {
            const auto kp = ptMatched[i];
            const auto kp3d = T * (camIntrinsics.inverse() * Vec3d(kp.pt.x / 640.0, kp.pt.y / 480.0, 0.01));
            // ptMatched3d.push_back(kp3d.cast<float>());
            // lines.push_back(ptMatchedDst3d[i]);
            // lines.push_back(ptMatched3d[i]);
        }

        // display matched points in 3d
        // viewer->setMatchedPoints(ptMatchedDst3d);
        // viewer->setMatchedFramePoints(ptMatched3d);
        // viewer->setMatchingLines(lines);
    }

    // display pose proposals
    if (viewerEnabled && viewer)
        viewer->setRelocalizationHypotheses(hypothesesList);

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
    lastTrackedPose = SE3(Mat44d::Identity());
    accumulateTransform = SE3(Mat44d::Identity());
    state = SystemState::NotInitialized;
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
    this->localOptimizer->setViewer(viewer);
}

void FullSystem::setMappingEnable(const bool enable)
{
    mappingEnabled = enable;
}

void FullSystem::setSystemStateToLost()
{
    state = SystemState::Lost;
}

void FullSystem::setSystemStateToTest()
{
    state = SystemState::Test;
}

void FullSystem::testNextKF()
{
    auto &KFs = map->getKeyframesAll();
    testKFId = (testKFId + 1) % KFs.size();
}

void FullSystem::setGraphMatching(const bool &flag)
{
    useGraphMatching = flag;
}

void FullSystem::setGraphGetNormal(const bool &flag)
{
    shouldCalculateNormal = flag;
}