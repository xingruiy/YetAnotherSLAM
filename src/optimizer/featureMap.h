#pragma once
#include <memory>
#include <mutex>
#include "utils/numType.h"
#include "utils/frame.h"
#include "mapViewer/mapViewer.h"
#include "optimizer/featureMatcher.h"

class FeatureMap
{
    Mat33d K, Kinv;
    bool shouldQuit;
    bool hasNewKF;
    bool isOptimizing;
    int optWinSize;
    MapViewer *viewer;

    std::shared_ptr<FeatureMatcher> matcher;

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
    // TODO: double check if two pts are the same one
    std::vector<std::shared_ptr<Frame>> findCloseLoopCandidate(std::shared_ptr<Frame> frame);
    void findPointCorrespondences(std::shared_ptr<Frame> kf, std::vector<std::shared_ptr<MapPoint>> mapPoints);

public:
    FeatureMap(Mat33d &K, int localWinSize = 5);

    void reset();
    void addFrameHistory(std::shared_ptr<Frame> frame);
    void addReferenceFrame(std::shared_ptr<Frame> frame);

    std::vector<SE3> getFrameHistory() const;
    std::vector<SE3> getKeyFrameHistory();
    std::vector<Vec3f> getActivePoints();
    std::vector<Vec3f> getStablePoints();

    void optimizationLoop();
    void globalConsistencyLoop();
    void setShouldQuit();
    bool hasUnfinishedWork();
    void setMapViewer(MapViewer *viewer);
};