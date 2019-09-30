#pragma once
#include <memory>
#include <mutex>
#include "utils/numType.h"
#include "utils/frame.h"
#include "mapViewer/mapViewer.h"
#include "optimizer/featureMatcher.h"

class FeatureMap
{
    MapViewer *viewer;
    bool optimizerRunning;
    bool updateMapView;
    size_t localWindowSize;

    std::mutex mutexKFBuffer;
    std::queue<std::shared_ptr<Frame>> keyframeBuffer;
    std::vector<std::shared_ptr<Frame>> keyframesAll;
    std::shared_ptr<Frame> lastReferenceKF;
    // Feature Matching
    std::shared_ptr<FeatureMatcher> matcher;
    // Loop Closure
    std::vector<std::shared_ptr<Frame>> findClosedCandidate(
        std::shared_ptr<Frame> frame,
        const float distTh,
        const bool checkFrustumOverlapping);
    std::vector<std::shared_ptr<Frame>> findDistantCandidate(
        std::shared_ptr<Frame> frame);
    // Bundle Adjustment
    void bundleAdjustmentAll(
        std::vector<std::shared_ptr<Frame>> kfs,
        std::vector<std::shared_ptr<MapPoint>> pts,
        const int maxIter);
    void bundleAdjustmentSubset(
        std::vector<std::shared_ptr<Frame>> kfs,
        std::vector<std::shared_ptr<MapPoint>> pts,
        const int maxIter);

    Mat33d K, Kinv;
    bool shouldQuit;
    bool hasNewKF;
    bool isOptimizing;
    int optWinSize;

    // use lock semantics. TODO: lock-free queues?
    std::mutex optWinMutex;
    std::mutex optimizerMutex;
    std::deque<std::shared_ptr<Frame>> keyframeOptWin;
    std::mutex historyMutex;
    std::vector<std::shared_ptr<Frame>> keyframeHistory;
    std::vector<std::pair<SE3, std::shared_ptr<Frame>>> frameHistory;
    std::mutex bufferMutex;
    std::queue<std::shared_ptr<Frame>> newKeyFrameBuffer;
    std::mutex loopBufferMutex;
    std::queue<std::shared_ptr<Frame>> loopKeyFrameBuffer;

    void addToOptimizer(std::shared_ptr<Frame> kf);
    void marginalizeOldFrame();
    void windowedOptimization(const int maxIter);

public:
    FeatureMap(Mat33d &K, int localWinSize = 5);
    FeatureMap(Mat33d &K, int localWinSize, bool updateView);
    void reset();
    void setMapViewer(MapViewer *viewer);
    void addFrameHistory(std::shared_ptr<Frame> frame);
    void addReferenceFrame(std::shared_ptr<Frame> frame);
    void localOptimizationLoop();
    void localOptimizationLoop2();

    void globalConsistencyLoop();
    void setShouldQuit();

    std::vector<SE3> getFrameHistory() const;
    std::vector<SE3> getKeyFrameHistory();
    std::vector<Vec3f> getActivePoints();
    std::vector<Vec3f> getStablePoints();
};