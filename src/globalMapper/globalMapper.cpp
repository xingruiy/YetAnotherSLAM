#include "globalMapper/globalMapper.h"
#include <ceres/ceres.h>

GlobalMapper::GlobalMapper(Mat33d &K, int localWinSize)
    : optWinSize(localWinSize), K(K),
      shouldQuit(false), hasNewKF(false)
{
    Kinv = K.inverse();
    solver = std::make_shared<CeresSolver>();
    matcher = std::make_shared<FeatureMatcher>(ORB);
}

void GlobalMapper::reset()
{
    frameHistory.clear();
    keyframeOptWin.clear();
    keyframeHistory.clear();
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
    kf2Del->cvKeyPoints.clear();
    kf2Del->pointDesc.clear();
    keyframeHistory.push_back(kf2Del);
    keyframeOptWin.pop_front();
}

void GlobalMapper::resetPointVisitFlag()
{
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

        frame->flagKeyFrame();
        std::cout << "processing: " << frame->kfId << std::endl;

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

            Mat outImg;
            Mat refImg = refKF->getImage();
            cv::drawMatches(refImg, refKF->cvKeyPoints, image, frame->cvKeyPoints, matches, outImg);
            cv::imshow("img", outImg);
            cv::waitKey(1);

            for (auto match : matches)
            {
                auto pt = refKF->mapPoints[match.queryIdx];
                auto framePt = frame->cvKeyPoints[match.trainIdx];
                auto obs = Vec2d(framePt.pt.x, framePt.pt.y);

                if (pt)
                {
                    frame->mapPoints[match.trainIdx] = pt;
                    pt->observations.push_back(std::pair<std::shared_ptr<Frame>, Vec2d>(frame, obs));
                }
            }
        }

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
                pt3d->position = frame->getPose() * (Kinv * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z);
                pt3d->descriptor = desc;
                frame->mapPoints[i] = pt3d;
            }
        }

        // add to frame history
        frameHistory.push_back(std::pair<SE3, std::shared_ptr<Frame>>(SE3(), frame));
        // add to optimization window
        keyframeOptWin.push_back(frame);

        optimizeWindow(15);
    }
}

void GlobalMapper::optimizeWindow(const int maxIteration)
{
}