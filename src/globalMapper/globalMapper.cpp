#include "globalMapper/globalMapper.h"
#include <ceres/ceres.h>

GlobalMapper::GlobalMapper(Mat33d &K, int localWinSize)
    : optWinSize(localWinSize), K(K),
      isOptimizing(false), shouldQuit(false), hasNewKF(false)
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

    std::unique_lock<std::mutex> lock(bufferMutex);
    newKeyFrameBuffer.push(frame);
}

void GlobalMapper::marginalizeOldFrame()
{
    std::shared_ptr<Frame> kf2Del = NULL;

    {
        std::unique_lock<std::mutex> lock(optWinMutex);
        kf2Del = keyframeOptWin.front();
        keyframeOptWin.pop_front();
    }

    for (auto pt : kf2Del->mapPoints)
    {
        if (pt && pt->inOptimizer && pt->hostKF == kf2Del)
        {
            if (pt->hostKF == kf2Del)
            {
                solver->removeWorldPoint(pt->ptId);
                pt->inOptimizer = false;
                pt->invalidated = true;
                if (pt->numObservations <= 1)
                    pt = NULL;
            }
            else
                solver->removeObservation(pt->ptId, kf2Del->getKeyframeId());
        }
    }

    solver->removeCamera(kf2Del->getKeyframeId());

    kf2Del->cvKeyPoints.clear();
    kf2Del->pointDesc.clear();

    {
        std::unique_lock<std::mutex> lock(historyMutex);
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
            if (pt && !pt->visited && !pt->invalidated && pt->inOptimizer)
            {
                pt->visited = true;
                localPoints.push_back(solver->getPtPosOptimized(pt->ptId));
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
            if (pt && !pt->visited && pt->numObservations > 1)
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

bool GlobalMapper::doubleCheckPointPair(Mat image, Mat refImage, cv::KeyPoint &pt, cv::KeyPoint &refPt)
{
    Mat desc, refDesc;
    matcher->compute(image, {pt}, desc);
    matcher->compute(refImage, {refPt}, refDesc);
    float score = matcher->computeMatchingScore(desc, refDesc);
    // std::cout << score << std::endl;
}

void GlobalMapper::optimizationLoop()
{
    while (!shouldQuit)
    {
        std::shared_ptr<Frame> frame = NULL;

        {
            std::unique_lock<std::mutex> lock(bufferMutex);
            if (newKeyFrameBuffer.size() > 0)
            {
                frame = newKeyFrameBuffer.front();
                newKeyFrameBuffer.pop();
            }
        }

        if (frame == NULL)
            continue;

        if (keyframeOptWin.size() >= optWinSize)
            marginalizeOldFrame();

        Mat image = frame->getImage();
        Mat depth = frame->getDepth();
        Mat intensity = frame->getIntensity();

        auto refKF = frame->getReferenceKF();
        if (refKF)
        {
            auto dT = frame->getTrackingResult();
            auto refT = refKF->getPoseInGlobalMap();
            auto T = refT * dT;
            frame->setOptimizationResult(T);
        }

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
            // matcher->matchByProjection(refKF, frame, K, matches, NULL);

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
                auto framePt3d = frame->mapPoints[match.trainIdx];
                auto framePt = frame->cvKeyPoints[match.trainIdx];
                auto obs = Vec2d(framePt.pt.x, framePt.pt.y);

                if (pt && !framePt3d)
                {
                    frame->mapPoints[match.trainIdx] = pt;
                    pt->numObservations++;
                }

                // if (pt && framePt3d && (pt != framePt3d))
                // {
                //     bool equal = doubleCheckPointPair(
                //         image,
                //         refKF->getImage(),
                //         frame->cvKeyPoints[match.trainIdx],
                //         refKF->cvKeyPoints[match.queryIdx]);
                // }
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
        {
            std::unique_lock<std::mutex> lock(historyMutex);
            frameHistory.push_back(std::pair<SE3, std::shared_ptr<Frame>>(SE3(), frame));
        }
        // add to optimization window
        {
            std::unique_lock<std::mutex> lock(optWinMutex);
            keyframeOptWin.push_back(frame);
        }

        addToOptimizer(frame);
        windowedOptimization(15);
    }
}

std::vector<std::shared_ptr<Frame>> GlobalMapper::findCloseLoopCandidate(std::shared_ptr<Frame> frame)
{
    auto framePose = frame->getPoseInGlobalMap();
    auto framePosition = framePose.translation();
    for (auto kf : keyframeHistory)
    {
    }
}

void GlobalMapper::globalConsistencyLoop()
{
    while (!shouldQuit)
    {
        std::shared_ptr<Frame> loopKF = NULL;
        if (loopKF == NULL)
            continue;
    }
}

void GlobalMapper::addToOptimizer(std::shared_ptr<Frame> kf)
{
    solver->addCamera(kf->getKeyframeId(), kf->getPoseInGlobalMap().inverse(), false);
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
    solver->optimize(maxIteration);

    {
        std::unique_lock<std::mutex> lock(optWinMutex);
        for (auto kf : keyframeOptWin)
        {
            for (auto pt : kf->mapPoints)
            {
                if (pt && pt->inOptimizer && !pt->invalidated)
                {
                    auto rval = solver->getPtPosOptimized(pt->ptId);
                    if (!rval.isApprox(Vec3f()))
                        pt->position = rval.cast<double>();
                }
            }

            auto poseOpt = solver->getCamPoseOptimized(kf->getKeyframeId()).inverse();
            kf->setOptimizationResult(poseOpt);
        }
    }
}

bool GlobalMapper::hasUnfinishedWork()
{
    std::unique_lock<std::mutex> lock(bufferMutex);
    return isOptimizing && newKeyFrameBuffer.size() > 0;
}

std::vector<SE3> GlobalMapper::getKeyFrameHistory()
{
    std::vector<SE3> history;

    {
        std::unique_lock<std::mutex> lock(historyMutex);
        for (auto kf : keyframeHistory)
        {
            history.push_back(kf->getPoseInGlobalMap());
        }
    }

    {
        std::unique_lock<std::mutex> lock(optWinMutex);
        for (auto kf : keyframeOptWin)
        {
            history.push_back(kf->getPoseInGlobalMap());
        }
    }

    return history;

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
    // return keyframePoseHistory;
}