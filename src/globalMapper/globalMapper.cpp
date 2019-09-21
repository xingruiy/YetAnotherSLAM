#include "globalMapper/globalMapper.h"
#include <ceres/ceres.h>

GlobalMapper::GlobalMapper(Mat33d &K, int localWinSize)
    : optWinSize(localWinSize), K(K),
      shouldQuit(false), hasNewKF(false)
{
    Kinv = K.inverse();
    solver = std::make_shared<CeresSolver>(K);
    matcher = std::make_shared<FeatureMatcher>(ORB);
}

void GlobalMapper::reset()
{
    frameHistory.clear();
    keyframeOptWin.clear();
    keyframeHistory.clear();
    keyframePoseHistory.clear();
}

void GlobalMapper::addFrameHistory(std::shared_ptr<Frame> frame)
{
    auto refKF = frame->getReferenceKF();
    if (refKF == NULL)
        return;

    frameHistory.push_back(std::make_pair(frame->getTrackingResult(), refKF));
}

void GlobalMapper::addReferenceFrame(std::shared_ptr<Frame> frame)
{
    if (frame == NULL)
        return;

    std::unique_lock<std::mutex> lock(localKeyFrameLock);
    newKeyFrameBuffer.push(frame);
}

void GlobalMapper::marginalizeOldFrame()
{
    std::shared_ptr<Frame> kf2Del = NULL;
    std::shared_ptr<Frame> secondOldestKF = NULL;

    {
        std::unique_lock<std::mutex> lock(localKeyFrameLock);
        kf2Del = keyframeOptWin.front();
        keyframeOptWin.pop_front();
        secondOldestKF = keyframeOptWin.front();
    }

    for (auto pt : kf2Del->mapPoints)
    {
        if (pt && pt->inOptimizer && pt->hostKF == kf2Del)
        {
            // printf("%lu camera and %lu pt\n", kf2Del->kfId, pt->ptId);
            solver->removeObservation(pt->ptId, kf2Del->getKeyframeId());
            if (pt->hostKF == kf2Del)
            {
                solver->removeWorldPoint(pt->ptId);
                pt->inOptimizer = false;
            }
        }

        // solver->removeCamera(kf2Del->kfId);
    }

    // solver->setCameraBlockConstant(secondOldestKF->kfId);

    kf2Del->cvKeyPoints.clear();
    kf2Del->pointDesc.clear();

    {
        std::unique_lock<std::mutex> lock(globalKeyFrameLock);
        keyframeHistory.push_back(kf2Del);
    }
}

std::vector<Vec3f> GlobalMapper::getActivePoints()
{
    std::vector<Vec3f> localPoints;

    for (auto kf : keyframeOptWin)
        if (kf)
            for (auto pt : kf->mapPoints)
                if (pt)
                    pt->visited = false;

    for (auto kf : keyframeOptWin)
    {
        if (kf == NULL)
            continue;

        for (auto pt : kf->mapPoints)
            if (pt && !pt->visited)
            {
                pt->visited = true;
                localPoints.push_back(pt->position.cast<float>());
            }
    }

    return localPoints;
}

std::vector<Vec3f> GlobalMapper::getStablePoints()
{
    std::vector<Vec3f> stablePoints;

    for (auto kf : keyframeHistory)
        if (kf)
            for (auto pt : kf->mapPoints)
                if (pt)
                    pt->visited = false;

    for (auto kf : keyframeHistory)
    {
        if (kf == NULL)
            continue;

        for (auto pt : kf->mapPoints)
            if (pt && !pt->visited)
            {
                pt->visited = true;
                stablePoints.push_back(pt->position.cast<float>());
            }
    }

    return stablePoints;
}

std::vector<SE3> GlobalMapper::getFrameHistory() const
{
    std::vector<SE3> history;
    for (auto &h : frameHistory)
    {
        auto &dT = h.first;
        auto &refKF = h.second;
        history.push_back(refKF->getPoseInGlobalMap() * dT);
    }

    return history;
}

void GlobalMapper::setShouldQuit()
{
    shouldQuit = true;
}

void GlobalMapper::optimizationLoop()
{
    while (!shouldQuit)
    {
        std::shared_ptr<Frame> frame = NULL;

        {
            std::unique_lock<std::mutex> lock(localKeyFrameLock);
            if (newKeyFrameBuffer.size() > 0)
            {
                frame = newKeyFrameBuffer.front();
                newKeyFrameBuffer.pop();
            }
        }

        if (frame == NULL)
            continue;

        // std::cout << "processing: " << frame->kfId << std::endl;

        if (keyframeOptWin.size() >= optWinSize)
            marginalizeOldFrame();

        Mat image = frame->getImage();
        Mat depth = frame->getDepth();
        Mat intensity = frame->getIntensity();

        std::vector<float> zVector;
        matcher->detect(image, depth, intensity, frame->cvKeyPoints, frame->pointDesc, zVector);
        int numDetectedPoints = frame->cvKeyPoints.size();

        if (numDetectedPoints == 0)
        {
            printf("Error: no features detected! keyframe not accepted.\n");
            return;
        }

        std::vector<bool> matchesFound(numDetectedPoints);
        std::fill(matchesFound.begin(), matchesFound.end(), false);
        frame->mapPoints.resize(numDetectedPoints);

        for (auto refKF : keyframeOptWin)
        {
            if (refKF == frame)
                continue;

            std::vector<cv::DMatch> matches;
            matcher->matchByProjection(refKF, frame, K, matches, &matchesFound);

            if (matches.size() == 0)
                continue;

            // Mat outImg;
            // Mat refImg = refKF->getImage();
            // cv::drawMatches(refImg, refKF->cvKeyPoints, image, frame->cvKeyPoints, matches, outImg);
            // cv::imshow("img", outImg);
            // cv::waitKey(1);

            for (auto match : matches)
            {
                auto pt = refKF->mapPoints[match.queryIdx];
                auto framePt = frame->cvKeyPoints[match.trainIdx];
                auto obs = Vec2d(framePt.pt.x, framePt.pt.y);

                if (pt)
                    frame->mapPoints[match.trainIdx] = pt;
            }
        }

        auto framePose = frame->getPoseInGlobalMap();
        for (int i = 0; i < numDetectedPoints; ++i)
        {
            if (matchesFound[i])
                continue;

            const auto &kp = frame->cvKeyPoints[i];
            const auto &desc = frame->pointDesc[i];
            const auto &z = zVector[i];

            if (z > FLT_EPSILON && desc.norm() > 0)
            {
                auto pt3d = std::make_shared<Point3D>();

                pt3d->hostKF = frame;
                pt3d->position = framePose * (Kinv * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z);
                pt3d->descriptor = desc;
                frame->mapPoints[i] = pt3d;
            }
        }

        // add to frame history
        frameHistory.push_back(std::pair<SE3, std::shared_ptr<Frame>>(SE3(), frame));
        // add to optimization window
        keyframeOptWin.push_back(frame);

        addToOptimizer(frame);
        windowedOptimization(15);
    }
}

void GlobalMapper::addToOptimizer(std::shared_ptr<Frame> kf)
{
    solver->addCamera(kf->getKeyframeId(), kf->getPoseInGlobalMap().inverse(), kf->getKeyframeId() == 0);
    for (int i = 0; i < kf->cvKeyPoints.size(); ++i)
    {
        auto pt = kf->mapPoints[i];
        if (pt)
        {
            const auto &kp = kf->cvKeyPoints[i];

            if (!pt->inOptimizer)
            {
                solver->addWorldPoint(pt->ptId, pt->position, true);
                pt->inOptimizer = true;
            }

            solver->addObservation(pt->ptId, kf->getKeyframeId(), Vec2d(kp.pt.x, kp.pt.y));
        }
    }
}

void GlobalMapper::windowedOptimization(const int maxIteration)
{
    size_t oldestKFId = 0;
    size_t newestKFId = 0;

    {
        std::unique_lock<std::mutex> lock(localKeyFrameLock);
        oldestKFId = keyframeOptWin.front()->getKeyframeId();
        newestKFId = keyframeOptWin.back()->getKeyframeId();
    }

    solver->optimize(maxIteration, oldestKFId, newestKFId);
    auto newestPose = solver->getCamera(newestKFId);
    keyframePoseHistory.push_back(newestPose);
}

std::vector<SE3> GlobalMapper::getKeyFrameHistory()
{
    // std::vector<std::shared_ptr<Frame>> copyHistory;

    // {
    //     std::unique_lock<std::mutex> lock(globalKeyFrameLock);
    //     copyHistory = keyframeHistory;
    // }

    // std::vector<SE3> history;
    // for (int i = 0; i < copyHistory.size(); ++i)
    // {
    //     auto kf = copyHistory[i];
    //     SE3 kfPose = SE3();

    //     if (solver->hasCamera(kf->kfId))
    //         kfPose = solver->getCamera(kf->kfId);
    //     else
    //         kfPose = kf->getPose();

    //     history.push_back(kfPose);
    // }

    // return history;
    return keyframePoseHistory;
}