#include "globalMapper/globalMapper.h"
#include <ceres/ceres.h>

GlobalMapper::GlobalMapper(Mat33d &K, int localWinSize)
    : optWinSize(localWinSize), K(K),
      shouldQuit(false), hasNewKF(false)
{
    Kinv = K.inverse();
    matcher = std::make_shared<FeatureMatcher>(FAST);
}

void GlobalMapper::reset()
{
    frameHistory.clear();
    keyframeOptWin.clear();
    keyframeHistory.clear();
    salientPointHistory.clear();
}

void GlobalMapper::addFrameHistory(const SE3 &T)
{
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
    auto kf2Del = keyframeOptWin.front();
    keyframeOptWin.pop_front();
    keyframeHistory.push_back(kf2Del);
}

std::vector<Vec3f> GlobalMapper::getActivePoints()
{
    std::vector<Vec3f> localPoints;

    for (auto kf : keyframeOptWin)
        if (kf)
            for (auto pt : kf->getWorldPoints())
                pt->visited = false;

    for (auto kf : keyframeOptWin)
    {
        if (kf == NULL)
            continue;

        auto framePoints = kf->getWorldPoints();
        for (auto pt : framePoints)
            if (!pt->visited)
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
            for (auto pt : kf->getWorldPoints())
                pt->visited = false;

    for (auto kf : keyframeHistory)
    {
        if (kf == NULL)
            continue;

        auto framePoints = kf->getWorldPoints();
        for (auto pt : framePoints)
            if (!pt->visited)
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
    }

    return history;
}

std::vector<Vec3f> GlobalMapper::getPointHistory() const
{
    return salientPointHistory;
}

std::vector<KFConstraint> GlobalMapper::searchConstraints(std::shared_ptr<Frame> frame)
{
    std::vector<KFConstraint> constraints;
    for (auto refKF : keyframeOptWin)
    {
        if (refKF == frame)
            continue;

        auto kpSize = refKF->getKeyPointSize();
        if (kpSize == 0)
            continue;

        std::vector<cv::DMatch> matches;
        SE3 transform = frame->getPose().inverse() * refKF->getPose();
        matcher->matchByProjection(refKF, frame, transform, matches);

        if (matches.size() == 0)
            continue;

        KFConstraint cons;
        cons.referenceKF = refKF;
        cons.frame = frame;
        cons.featureCorresp = matches;
        constraints.push_back(cons);
    }

    return constraints;
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

        if (keyframeOptWin.size() >= optWinSize)
            marginalizeOldFrame();

        Mat image = frame->getImage();
        Mat depth = frame->getDepth();
        Mat intensity = frame->getIntensity();

        std::vector<cv::KeyPoint> keyPoints;
        std::vector<Vec9f> localPatch;
        std::vector<float> zVector;

        matcher->detect(image, depth, intensity, keyPoints, localPatch, zVector);
        int numDetectedPoints = keyPoints.size();

        if (numDetectedPoints == 0)
        {
            printf("Error: no features detected! keyframe not accepted.\n");
            return;
        }

        std::vector<bool> matchMask(numDetectedPoints);
        std::fill(matchMask.begin(), matchMask.end(), true);

        for (auto refKF : keyframeOptWin)
        {
            if (refKF == frame)
                continue;

            SE3 transform = refKF->getPose().inverse() * frame->getPose();
            std::vector<cv::DMatch> matches;
            matcher->matchByProjection(refKF, frame, transform, matches);

            if (matches.size() == 0)
                continue;

            auto referencePoints = refKF->getWorldPoints();

            for (auto match : matches)
            {
                auto refIdx = match.trainIdx;
                auto fIdx = match.queryIdx;

                if (!matchMask[fIdx])
                    continue;

                auto refPt = referencePoints[refIdx];
                auto framePt = keyPoints[fIdx];
                auto obs = Vec2d(framePt.pt.x, framePt.pt.y);
                refPt->observations.push_back(std::pair<std::shared_ptr<Frame>, Vec2d>(frame, obs));

                matchMask[fIdx] = false;
            }
        }

        std::vector<std::shared_ptr<PointWorld>> pointsWorld(numDetectedPoints);

        for (int i = 0; i < numDetectedPoints; ++i)
        {
            if (!matchMask[i])
                continue;

            const auto &kp = keyPoints[i];
            const auto &desc = localPatch[i];
            const auto &z = zVector[i];

            if (z > FLT_EPSILON && desc.norm() > 0)
            {
                Vec3d pt = Kinv * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z;
                Vec3d ptWorld = frame->getPose() * pt;

                auto ptWd = std::make_shared<PointWorld>();
                ptWd->hostKF = frame;
                ptWd->position = ptWorld;
                ptWd->descriptor = desc;
                pointsWorld[i] = ptWd;
            }
        }

        for (auto refKF : keyframeOptWin)
        {
            if (refKF == frame)
                continue;

            SE3 transform = frame->getPose().inverse() * refKF->getPose();
            std::vector<cv::DMatch> matches;
            matcher->matchByProjection(frame, refKF, transform, matches, &matchMask);

            if (matches.size() == 0)
                continue;

            for (auto match : matches)
            {
                auto refIdx = match.trainIdx;
                auto fIdx = match.queryIdx;
            }
        }

        frame->setWorldPoints(pointsWorld);

        // add to frame history
        frameHistory.push_back(std::pair<SE3, std::shared_ptr<Frame>>(SE3(), frame));

        // auto constraints = searchConstraints(frame);
        keyframeOptWin.push_back(frame);
    }
}

void GlobalMapper::optimizeWindow(const int maxIteration)
{
    auto camera = Vec4d(K(0, 0), K(1, 1), K(0, 2), K(1, 2));
    double *cameraBlock = (double *)Eigen::internal::aligned_malloc(sizeof(double) * 30);

    for (int k = 0; k < optWinSize; ++k)
    {
        SE3 pose = keyframeOptWin[k]->getPose().inverse();
        for (int i = 0; i < 6; ++i)
            cameraBlock[k * 6 + i] = pose.data()[i];
    }

    ceres::Problem system;
    Eigen::internal::aligned_free(cameraBlock);
}