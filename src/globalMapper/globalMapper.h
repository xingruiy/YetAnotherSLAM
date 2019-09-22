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

    std::vector<SE3> keyframePoseHistory;
    std::shared_ptr<FeatureMatcher> matcher;
    std::deque<std::shared_ptr<Frame>> keyframeOptWin;
    std::vector<std::shared_ptr<Frame>> keyframeHistory;

    std::mutex localKeyFrameLock;
    std::mutex globalKeyFrameLock;
    std::queue<std::shared_ptr<Frame>> newKeyFrameBuffer;
    std::vector<std::pair<SE3, std::shared_ptr<Frame>>> frameHistory;

    void addToOptimizer(std::shared_ptr<Frame> kf);
    void marginalizeOldFrame();

    std::shared_ptr<CeresSolver> solver;

    bool doubleCheckPointPair(Mat image, Mat refImage, cv::KeyPoint &pt, cv::KeyPoint &refPt);

public:
    GlobalMapper(Mat33d &K, int localWinSize = 5);

    void reset();
    void addFrameHistory(std::shared_ptr<Frame> frame);
    void addReferenceFrame(std::shared_ptr<Frame> frame);

    std::vector<SE3> getFrameHistory() const;
    std::vector<SE3> getKeyFrameHistory();
    std::vector<Vec3f> getActivePoints();
    std::vector<Vec3f> getStablePoints();

    void optimizationLoop();
    void setShouldQuit();
    void windowedOptimization(const int maxIter);
};