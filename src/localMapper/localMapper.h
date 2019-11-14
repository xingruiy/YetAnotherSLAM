#pragma once
#include <memory>
#include <mutex>
#include "dataStruct/map.h"
#include "utils/numType.h"
#include "dataStruct/keyFrame.h"
#include "mapViewer/mapViewer.h"
#include "localMapper/featureMatcher.h"

class LocalMapper
{
public:
    LocalMapper(const Mat33d &K);

    inline void setMap(Map *map)
    {
        this->map = map;
    }

    inline void setMapViewer(MapViewer *viewer)
    {
        this->viewer = viewer;
    }

    void addKeyFrame(std::shared_ptr<KeyFrame> KF);
    void loop();

    inline void setShouldQuit()
    {
        shouldQuit = true;
    }

private:
    Mat33d K;
    Map *map;
    MapViewer *viewer;
    bool shouldQuit;

    // std::shared_ptr<Map> map;
    std::shared_ptr<FeatureMatcher> matcher;

    inline bool hasNewKeyFrame()
    {
        std::unique_lock<std::mutex> lock(mutexKeyFrameQueue);
        return !keyFrameQueue.empty();
    }

    void processNewKeyFrame();
    int matchLocalMapPoints();
    void createNewMapPoints();
    void updateLocalKeyFrames();
    void updateLocalMapPoints();
    void createInitMapPoints();
    void optimizeKeyFramePose();

    std::shared_ptr<KeyFrame> lastKeyFrame;
    std::shared_ptr<KeyFrame> currKeyFrame;
    std::mutex mutexKeyFrameQueue;
    std::deque<std::shared_ptr<KeyFrame>> keyFrameQueue;

    std::vector<std::shared_ptr<KeyFrame>> localKeyFrameSet;
    std::vector<std::shared_ptr<MapPoint>> localMapPointSet;

    // void matchFeatures(std::shared_ptr<Frame> kf);
    // void detectLoop(std::shared_ptr<Frame> kf);
    // void createNewPoints(std::shared_ptr<Frame> kf);
    // std::shared_ptr<Frame> getNewKeyframe();
};