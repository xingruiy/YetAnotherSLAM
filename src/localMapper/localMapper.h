#pragma once
#include <memory>
#include <mutex>
#include "dataStruct/map.h"
#include "utils/numType.h"
#include "mapViewer/mapViewer.h"
#include "localMapper/featureMatcher.h"

class LocalMapper
{
public:
    LocalMapper(Mat33d &K, std::shared_ptr<Map> map, MapViewer &viewer);
    void addKeyFrame(std::shared_ptr<Frame> keyFrame);

    void loop();
    void setShouldQuit();

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

    bool hasNewKeyFrame();
    void checkKeyFramePose();
    void processNewKeyFrame();
    void createNewMapPoints();

    std::shared_ptr<Frame> lastKeyFrame;
    std::shared_ptr<Frame> currKeyFrame;
    std::deque<std::shared_ptr<Frame>> newKeyFrames;
    std::mutex newKeyFrameMutex;
};