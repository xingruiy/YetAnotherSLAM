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

    void optimize(
        std::vector<std::shared_ptr<Frame>> kfs,
        std::vector<std::shared_ptr<MapPoint>> pts,
        const int maxIter);

    void optimize(std::shared_ptr<Frame> kf);
    void optimizePoints(std::shared_ptr<Frame> kf);

    Mat33d K;
    bool shouldQuit;

public:
    LocalOptimizer(
        Mat33d &K,
        int localWinSize,
        std::shared_ptr<Map> map);

    void loop();
    void reset();
    void setShouldQuit();
    bool pauseMapping;
    void setMap(std::shared_ptr<Map> map);
};