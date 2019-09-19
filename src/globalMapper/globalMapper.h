#pragma once
#include <memory>
#include "utils/numType.h"
#include "utils/frame.h"
#include "featureMatcher.h"

class GlobalMapper
{
    Mat33d K, Kinv;
    const int optWinSize;

    std::vector<Vec3f> salientPointHistory;
    std::shared_ptr<FeatureMatcher> matcher;
    std::deque<std::shared_ptr<Frame>> keyframeOptWin;
    std::vector<std::shared_ptr<Frame>> keyframeHistory;
    std::vector<std::pair<SE3, std::shared_ptr<Frame>>> frameHistory;

    void marginalizeOldFrame();

public:
    GlobalMapper(Mat33d &K, int localWinSize = 5);

    void reset();
    void addFrameHistory(const SE3 &T);
    void addReferenceFrame(std::shared_ptr<Frame> frame);
    std::vector<Vec3f> getActivePoints();
    std::vector<Vec3f> getStablePoints();
    std::vector<Vec3f> getPointHistory() const;
    std::vector<SE3> getFrameHistory() const;
};