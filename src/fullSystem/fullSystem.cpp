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
      lastState(SystemState::NotInitialized),
      enableLocalMapping(true)
{
    resetStatistics();

    map = std::make_shared<Map>();
    denseMapper = std::make_shared<DenseMapping>(w, h, K);

    localMapper = std::make_shared<LocalMapper>(K);
    localMapper->setMap(map.get());
    localMapper->setMapViewer(&viewer);

    coarseTracker = std::make_shared<DenseTracker>(w, h, K, numLvl);

    lastTrackedPose = SE3(Mat44d::Identity());
    rawTransformation = SE3(Mat44d::Identity());

    gpuBufferVec4FloatWxH.create(h, w, CV_32FC4);
    gpuBufferFloatWxH.create(h, w, CV_32FC1);

    this->viewer = &viewer;
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
    currentFrame = Frame(imRGB, imDepth, cpuBufferFloatWxH, cpuBufferVec4FloatWxH, K);

    switch (state)
    {
    case SystemState::NotInitialized:
    {
        coarseTracker->setReferenceFrame(currentFrame);

        createNewKeyFrame();
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
                createNewKeyFrame();
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
        mappingEnabled = false;
        if (trackCurrentFrame())
        {
            fuseCurrentFrame();
            raytraceCurrentFrame();
        }

        // Update only when testKFId changed
        if (lastState == SystemState::Test && testKFId == lastTestedKFId)
            break;

        if (tryRelocalizeCurrentFrame())
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
    coarseTracker->setTrackingFrame(currentFrame);
    SE3 tRes = coarseTracker->getIncrementalTransform();
    // accumulated local transform
    rawTransformation = rawTransformation * tRes.inverse();

    if (viewer)
    {
        viewer->setCurrentCamera(lastTrackedPose * rawTransformation);
    }

    return true;
}

void FullSystem::fuseCurrentFrame()
{
    // Update dense map
    auto Rt = lastTrackedPose * rawTransformation;
    auto currDepth = coarseTracker->getReferenceDepth();
    denseMapper->fuseFrame(currDepth, Rt);
}

void FullSystem::raytraceCurrentFrame()
{
    // Update vmap and inverse depth map
    auto Rt = lastTrackedPose * rawTransformation;
    denseMapper->raytrace(gpuBufferVec4FloatWxH, Rt);
    coarseTracker->setReferenceInvDepth(gpuBufferVec4FloatWxH);

    // Compute and update nmap
    computeNormal(gpuBufferVec4FloatWxH, gpuBufferVec4FloatWxH2);
    gpuBufferVec4FloatWxH2.download(currentFrame.nmap);
}

bool FullSystem::tryRelocalizeKeyframe(std::shared_ptr<Frame> kf)
{
}

bool FullSystem::tryRelocalizeCurrentFrame()
{
    Mat descriptor;
    std::vector<cv::KeyPoint> keyPoints;

    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detect(currentFrame.imRGB, keyPoints);
    detector->compute(currentFrame.imRGB, keyPoints, descriptor);

    if (viewer)
    {
        cv::drawKeypoints(currentFrame.imRGB, keyPoints, cpuBufferVec3ByteWxH, cv::Scalar(255, 0, 0));
        viewer->setKeyPointImage(cpuBufferVec3ByteWxH);
    }

    auto nPoints = keyPoints.size();

    if (nPoints < 10)
    {
        printf("Too few points detected(%lu), relocalization failed...\n", nPoints);
        return false;
    }

    std::vector<Vec3d> x3DPoints(nPoints);
    std::vector<Vec3f> x3DNormal(nPoints);
    std::vector<bool> inliers(nPoints);
    std::fill(inliers.begin(), inliers.end(), true);

    for (int i = 0; i < nPoints; ++i)
    {
        auto &kp = keyPoints[i];
        const auto &x = kp.pt.x;
        const auto &y = kp.pt.y;
        auto &z = currentFrame.imDepth.ptr<float>((int)round(y))[(int)round(x)];
        auto &n = cpuBufferVec4FloatWxH.ptr<Vec3f>((int)round(y))[(int)round(x)];

        if (z > FLT_EPSILON && n(0) > FLT_EPSILON)
        {
            x3DPoints[i] = K.inverse() * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z;
            x3DNormal[i] = n;
        }
        else
            inliers[i] = false;
    }

    Localizer relocalizer;
    std::vector<SE3> hypothesesList;
    std::vector<std::vector<bool>> filter;
    std::vector<std::vector<cv::DMatch>> matches;

    // std::vector<std::vector<cv::DMatch>> rawMatches;
    // std::vector<cv::DMatch> refinedMatches;
    // auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING2);
    // matcher->knnMatch(map->descriptorDB, descriptor, rawMatches, 2);
    // const double ratioTest = 0.7;

    if (!relocalizer.getRelocHypotheses(
            map,
            x3DPoints,
            x3DNormal,
            descriptor,
            inliers,
            hypothesesList,
            matches,
            filter,
            useGraphMatching))
        return false;

    if (hypothesesList.size() == 0)
    {
        printf("Too few hypotheses(%lu), relocalization failed...\n", hypothesesList.size());
        return false;
    }

    // display matched points in the image
    if (matches.size() != 0 && filter.size() != 0)
    {
        cv::Mat test;

        std::vector<cv::KeyPoint> ptMatched;
        std::vector<cv::KeyPoint> ptMatchedDst;
        std::vector<cv::DMatch> ptMatch;
        std::vector<cv::Point> matchingLines;

        auto &mapPoints = map->mapPointDB;
        const auto &mlist = matches[0];
        const auto &outlier = filter[0];
        const auto &RTinv = hypothesesList[0].inverse();
        const auto fx = K(0, 0);
        const auto fy = K(1, 1);
        const auto cx = K(0, 2);
        const auto cy = K(1, 2);

        int nMapPoints = 0;

        for (auto i = 0; i < mlist.size(); ++i)
        {
            if (!outlier[i])
            {
                const auto &m = mlist[i];
                const auto &pos = RTinv * mapPoints[m.trainIdx]->pos;
                const int x = fx * pos(0) / pos(2) + cx;
                const int y = fy * pos(1) / pos(2) + cy;
                if (x >= 0 && y >= 0 && x < 640 && y < 480)
                {
                    ptMatched.push_back(keyPoints[m.queryIdx]);
                    ptMatchedDst.push_back(cv::KeyPoint(x, y, 1));
                    ptMatch.push_back(cv::DMatch(nMapPoints, nMapPoints, 0));
                    matchingLines.push_back(cv::Point2f(x, y));
                    matchingLines.push_back(keyPoints[m.queryIdx].pt);
                    nMapPoints++;
                }
            }
        }

        std::vector<cv::KeyPoint> vcMapPoints;
        for (auto &mp : mapPoints)
        {
            if (mp)
            {
                const auto &pos = RTinv * mp->pos;
                const int x = fx * pos(0) / pos(2) + cx;
                const int y = fy * pos(1) / pos(2) + cy;
                if (x >= 0 && y >= 0 && x < 640 && y < 480)
                {
                    vcMapPoints.push_back(cv::KeyPoint(x, y, 1));
                }
            }
        }

        // display matched points in a image;
        cv::drawKeypoints(currentFrame.imRGB, vcMapPoints, cpuBufferVec3ByteWxH, cv::Scalar(0, 0, 255));
        cv::drawKeypoints(cpuBufferVec3ByteWxH, ptMatched, cpuBufferVec3ByteWxH, cv::Scalar(0, 255, 0));
        cv::drawKeypoints(cpuBufferVec3ByteWxH, ptMatchedDst, test, cv::Scalar(255, 0, 0));
        for (int i = 0; i < nMapPoints - 1; ++i)
        {
            cv::line(test, matchingLines[2 * i], matchingLines[2 * i + 1], cv::Scalar(0, 0, 255));
        }

        if (viewer)
            viewer->setMatchedPointImage(test);
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

void FullSystem::createNewKeyFrame()
{
    currKeyFrame = std::make_shared<KeyFrame>(currentFrame);

    if (lastKeyFrame)
    {
        currKeyFrame->parent = lastKeyFrame;
        currKeyFrame->setPose(lastKeyFrame->RT * rawTransformation);
    }

    if (enableLocalMapping)
    {
        localMapper->addKeyFrame(currKeyFrame);
    }

    // Updtae local dense map pose
    lastTrackedPose = lastTrackedPose * rawTransformation;

    if (viewer)
    {
        viewer->addRawKeyFramePose(lastTrackedPose);
    }

    rawTransformation = SE3();
    lastKeyFrame = currKeyFrame;
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
}

std::vector<SE3> FullSystem::getFramePoseHistory()
{
}

std::vector<Vec3f> FullSystem::getMapPointPosAll()
{
    std::vector<Vec3f> mapPoints;
    map->getMapPoint(mapPoints);
    return mapPoints;
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
    testKFId++;
}

void FullSystem::setGraphMatching(const bool &flag)
{
    useGraphMatching = flag;
}

void FullSystem::setGraphGetNormal(const bool &flag)
{
    shouldCalculateNormal = flag;
}