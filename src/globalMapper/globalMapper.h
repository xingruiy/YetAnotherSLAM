#pragma once
#include <memory>
#include <mutex>
#include "utils/numType.h"
#include "utils/frame.h"
#include "globalMapper/featureMatcher.h"
#include "globalMapper/ceresSolver.h"

class GlobalMapper
{
    Mat33d K, Kinv;
    const int optWinSize;
    bool shouldQuit;
    bool hasNewKF;

    std::shared_ptr<FeatureMatcher> matcher;
    std::deque<std::shared_ptr<Frame>> keyframeOptWin;
    std::vector<std::shared_ptr<Frame>> keyframeHistory;

    std::mutex localKeyFrameLock;
    std::queue<std::shared_ptr<Frame>> newKeyFrameBuffer;
    std::vector<std::pair<SE3, std::shared_ptr<Frame>>> frameHistory;

    void marginalizeOldFrame();

    std::shared_ptr<CeresSolver> solver;

public:
    GlobalMapper(Mat33d &K, int localWinSize = 5);

    void reset();
    void addFrameHistory(const SE3 &T);
    void addReferenceFrame(std::shared_ptr<Frame> frame);

    void resetPointVisitFlag();
    std::vector<Vec3f> getActivePoints();
    std::vector<Vec3f> getStablePoints();
    std::vector<SE3> getFrameHistory() const;
    void optimizationLoop();
    void setShouldQuit();
    void optimizeWindow(const int maxIteration);
};