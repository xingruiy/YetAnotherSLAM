#include "globalMapper/globalMapper.h"

GlobalMapper::GlobalMapper(Mat33d &K, int localWinSize)
    : optWinSize(localWinSize), K(K)
{
    Kinv = K.inverse();
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

    for (int i = 0; i < numDetectedPoints; ++i)
    {
        const auto &kp = keyPoints[i];
        const auto &desc = localPatch[i];
        const auto &z = zVector[i];

        if (z > FLT_EPSILON && desc.norm() > 0)
        {
            Vec3d pt = Kinv * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z;
            Vec3d ptWorld = frame->getPose() * pt;
            salientPointHistory.push_back(ptWorld.cast<float>());
        }
    }

    // add to frame history
    frameHistory.push_back(std::pair<SE3, std::shared_ptr<Frame>>(SE3(), frame));

    if (keyframeOptWin.size() >= optWinSize)
        marginalizeOldFrame();

    keyframeOptWin.push_back(frame);
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
    {
        if (kf == NULL)
            continue;
    }

    return localPoints;
}

std::vector<Vec3f> GlobalMapper::getStablePoints()
{
    std::vector<Vec3f> stablePoints;
    for (auto kf : keyframeHistory)
    {
        if (kf == NULL)
            continue;
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