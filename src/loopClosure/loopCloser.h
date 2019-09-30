#pragma once
#include <vector>
#include <memory>
#include <ceres/ceres.h>
#include "utils/map.h"

class LoopCloser
{
    Mat33d K;
    bool shouldQuit;
    std::shared_ptr<Map> map;

    void optimize(
        std::vector<std::shared_ptr<Frame>> kfs,
        std::vector<std::shared_ptr<MapPoint>> pts,
        const int maxIter);

public:
    LoopCloser(const Mat33d &K, std::shared_ptr<Map> map);
    void loop();
    void setShouldQuit();
    void setMap(std::shared_ptr<Map> map);
};