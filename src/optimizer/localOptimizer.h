#pragma once
#include <memory>
#include <mutex>
#include "utils/map.h"
#include "utils/numType.h"
#include "mapViewer/mapViewer.h"
#include "optimizer/featureMatcher.h"

class LocalOptimizer
{
    std::shared_ptr<Map> map;
    std::shared_ptr<FeatureMatcher> matcher;

    void optimize(std::shared_ptr<Frame> kf);
    void optimizePoints(std::shared_ptr<Frame> kf);
    void optimize(
        std::vector<std::shared_ptr<Frame>> kfs,
        std::vector<std::shared_ptr<MapPoint>> pts,
        const int maxIter);

    void matchFeatures(std::shared_ptr<Frame> kf);
    void detectLoop(std::shared_ptr<Frame> kf);
    void createNewPoints(std::shared_ptr<Frame> kf);
    std::shared_ptr<Frame> getNewKeyframe();

    Mat33d K;
    MapViewer *viewer;
    bool shouldQuit;

public:
    LocalOptimizer(
        Mat33d &K,
        int localWinSize,
        std::shared_ptr<Map> map);

    void loop();
    void setShouldQuit();
    bool pauseMapping;
    void setViewer(MapViewer *viewer);
    void setMap(std::shared_ptr<Map> map);
};