#pragma once
#include <memory>
#include <mutex>
#include "dataStruct/map.h"
#include "utils/numType.h"
#include "mapViewer/mapViewer.h"
#include "localMapper/featureMatcher.h"

class FeatureMapper
{
public:
    FeatureMapper(Mat33d &K, std::shared_ptr<Map> map);
    void loop();
    void setShouldQuit();
    void setViewer(MapViewer *viewer);
    void setMap(std::shared_ptr<Map> map);

private:
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

    std::shared_ptr<Map> map;
    std::shared_ptr<FeatureMatcher> matcher;
};