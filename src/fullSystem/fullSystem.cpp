#include "fullSystem/fullSystem.h"
#include "localMapper/localizer.h"
#include "denseTracker/cudaImageProc.h"

FullSystem::FullSystem(
    int w, int h,
    Mat33d K,
    int numLvl,
    MapViewer &viewer)
    : K(K),
      testKFId(0),
      lastTestedKFId(0),
      mappingEnabled(true),
      useGraphMatching(false),
      shouldCalculateNormal(false),
      state(SystemState::NotInitialized),
      lastState(SystemState::NotInitialized)
{
    resetStatistics();

    map = std::make_shared<Map>();
    this->viewer = &viewer;
    denseMapper = std::make_shared<DenseMapping>(w, h, K);
    localMapper = std::make_shared<LocalMapper>(K, map, viewer);
    coarseTracker = std::make_shared<DenseTracker>(w, h, K, numLvl);

    lastTrackedPose = SE3(Mat44d::Identity());
    rawTransformation = SE3(Mat44d::Identity());

    gpuBufferVec4FloatWxH.create(h, w, CV_32FC4);
    gpuBufferFloatWxH.create(h, w, CV_32FC1);

    localMappingThread = std::thread(&LocalMapper::loop, localMapper.get());
}

FullSystem::~FullSystem()
{
    localMapper->setShouldQuit();
    printf("wating other threads to finish...\n");
    localMappingThread.join();
    printf("all threads finished!\n");
}

void FullSystem::resetStatistics()
{
    numFramesProcessed = 0;
    numRelocalisationAttempted = 0;
}

void FullSystem::setCpuBufferVec4FloatWxH(Mat buffer)
{
    cpuBufferVec4FloatWxH = buffer;
}

void FullSystem::processFrame(Mat imRGB, Mat imDepth)
{
    imRGB.convertTo(cpuBufferVec3FloatWxH, CV_32FC3);
    cv::cvtColor(cpuBufferVec3FloatWxH, cpuBufferFloatWxH, cv::COLOR_RGB2GRAY);
    currentFrame = std::make_shared<Frame>(imRGB, imDepth, cpuBufferFloatWxH, K);

    switch (state)
    {
    case SystemState::NotInitialized:
    {
        coarseTracker->setReferenceFrame(*currentFrame);

        createNewKF();
        fuseCurrentFrame();
        raytraceCurrentFrame();

        state = SystemState::OK;

        if (viewer)
            viewer->setCurrentState(0);

        break;
    }

    case SystemState::OK:
    {
        if (trackCurrentFrame())
        {
            fuseCurrentFrame();
            raytraceCurrentFrame();

            if (needNewKF())
                createNewKF();
        }
        else
        {
            state = SystemState::Lost;
        }

        break;
    }

    case SystemState::Lost:
    {
        printf("tracking loast, attempt to resuming...\n");
        tryRelocalizeCurrentFrame();

        break;
    }

    case SystemState::Test:
    {
        // Update only when testKFId changed
        if (lastState == SystemState::Test && testKFId == lastTestedKFId)
            break;

        printf("testing kf: %lu / %lu\n", testKFId, map->keyFrameDB.size());
        auto &testKF = map->keyFrameDB[testKFId];

        if (tryRelocalizeKeyframe(testKF))
            lastTestedKFId = testKFId;

        break;
    }
    }

    // Copy state
    lastState = state;

    // Update statistics
    if (state == SystemState::OK)
        numFramesProcessed += 1;
}

bool FullSystem::trackCurrentFrame()
{
    coarseTracker->setTrackingFrame(*currentFrame);
    SE3 tRes = coarseTracker->getIncrementalTransform();
    // accumulated local transform
    rawTransformation = rawTransformation * tRes.inverse();
    currentFrame->setTrackingResult(rawTransformation);
    currentFrame->setReferenceKF(currentKeyframe);

    return true;
}

void FullSystem::fuseCurrentFrame()
{
    auto Rt = lastTrackedPose * rawTransformation;
    auto currDepth = coarseTracker->getReferenceDepth();
    denseMapper->fuseFrame(currDepth, Rt);
}

void FullSystem::raytraceCurrentFrame()
{
    auto Rt = lastTrackedPose * rawTransformation;
    denseMapper->raytrace(gpuBufferVec4FloatWxH, Rt);
    coarseTracker->setReferenceInvDepth(gpuBufferVec4FloatWxH);
    computeNormal(gpuBufferVec4FloatWxH, gpuBufferVec4FloatWxH2);
    gpuBufferVec4FloatWxH2.download(currentFrame->nmap);
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
    if (viewer)
    {
        cv::drawKeypoints(kf->getImage(), cvKeyPoints, cpuBufferVec3ByteWxH, cv::Scalar(255, 0, 0));
        viewer->setKeyPointImage(cpuBufferVec3ByteWxH);
    }

    auto numFeatures = kf->cvKeyPoints.size();
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
            keyPoint[n] = K.inverse() * Vec3d(kp.x, kp.y, 1.0) * z;
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
        viewer)
    {
        std::vector<cv::KeyPoint> ptMatched;
        std::vector<cv::KeyPoint> ptMatchedDst;
        std::vector<cv::DMatch> ptMatch;
        std::vector<cv::Point> matchingLines;
        auto &mapPoints = map->mapPointDB;
        const auto &mlist = matches[0];
        const auto &outlier = filter[0];
        const auto &T = hypothesesList[0];
        const auto fx = K(0, 0);
        const auto fy = K(1, 1);
        const auto cx = K(0, 2);
        const auto cy = K(1, 2);
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

        viewer->setMatchedPointImage(test);
    }

    // display pose proposals
    if (viewer)
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
    if (viewer)
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
            keyPoint[n] = K.inverse() * Vec3d(kp.x, kp.y, 1.0) * z;
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
        viewer)
    {
        std::vector<cv::KeyPoint> ptMatched;
        // std::vector<Vec3f> ptMatched3d;
        // std::vector<Vec3f> ptMatchedDst3d;
        // std::vector<Vec3f> lines;
        auto &mapPoints = map->mapPointDB;
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

        auto cx = K(0, 2);
        auto cy = K(1, 2);
        for (int i = 0; i < ptMatched.size(); ++i)
        {
            const auto kp = ptMatched[i];
            const auto kp3d = T * (K.inverse() * Vec3d(kp.pt.x / 640.0, kp.pt.y / 480.0, 0.01));
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
    if (viewer)
        viewer->setRelocalizationHypotheses(hypothesesList);

    return true;
}

bool FullSystem::needNewKF()
{
    Vec3d t = rawTransformation.translation();
    if (t.norm() >= 0.3)
        return true;

    Vec3d r = rawTransformation.log().tail<3>();
    if (r.norm() >= 0.3)
        return true;

    return false;
}

void FullSystem::createNewKF()
{
    currentKeyframe = currentFrame;

    currentKeyframe->flagKeyFrame();
    lastTrackedPose = lastTrackedPose * rawTransformation;
    currentKeyframe->setRawKeyframePose(lastTrackedPose);

    lastKeyFrame = currKeyFrame;
    currKeyFrame = std::make_shared<KeyFrame>(*currentFrame);
    currKeyFrame->parent = lastKeyFrame;
    currKeyFrame->RT = lastTrackedPose;
    currKeyFrame->RTinv = lastTrackedPose.inverse();

    if (mappingEnabled)
    {
        map->addUnprocessedKeyframe(currentKeyframe);
        map->addKeyframePoseRaw(lastTrackedPose);
        // localMapper->addKeyFrame(currentKeyframe);
        map->addFramePose(SE3(), currentKeyframe);
    }

    if (viewer)
        viewer->addRawKeyFramePose(lastTrackedPose);

    rawTransformation = SE3();
}

void FullSystem::resetSystem()
{
    map->clear();
    viewer->resetViewer();
    denseMapper->reset();
    lastTrackedPose = SE3(Mat44d::Identity());
    rawTransformation = SE3(Mat44d::Identity());
    state = SystemState::NotInitialized;
    resetStatistics();
}

size_t FullSystem::getMesh(float *vbuffer, float *nbuffer, size_t bufferSize)
{
    return denseMapper->fetchMeshWithNormal(vbuffer, nbuffer);
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
    testKFId = (testKFId + 1) % map->keyFrameDB.size();
}

void FullSystem::setGraphMatching(const bool &flag)
{
    useGraphMatching = flag;
}

void FullSystem::setGraphGetNormal(const bool &flag)
{
    shouldCalculateNormal = flag;
}