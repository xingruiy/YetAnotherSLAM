#pragma once
#include <memory>
#include <mutex>
#include "Map.h"

#include "KeyFrame.h"
#include "Viewer.h"

class MapViewer;

class LocalMapper
{
public:
    LocalMapper(const Eigen::Matrix3d &K);

    inline void setMap(Map *map)
    {
        this->map = map;
    }

    inline void setMapViewer(MapViewer *viewer)
    {
        this->viewer = viewer;
    }

    void addKeyFrame(std::shared_ptr<KeyFrame> KF);
    void run();

    inline void setShouldQuit()
    {
        shouldQuit = true;
    }

    void enableMapping()
    {
        updateLocalMap = true;
    }

    void disableMapping()
    {
        updateLocalMap = false;
    }

private:
    Eigen::Matrix3d K;
    Map *map;
    MapViewer *viewer;
    bool shouldQuit;
    bool updateLocalMap;

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
    int checkMapPointOutliers();

    std::shared_ptr<KeyFrame> lastKeyFrame;
    std::shared_ptr<KeyFrame> currKeyFrame;
    std::mutex mutexKeyFrameQueue;
    std::deque<std::shared_ptr<KeyFrame>> keyFrameQueue;
    std::vector<std::shared_ptr<KeyFrame>> localKeyFrameSet;
    std::vector<std::shared_ptr<MapPoint>> localMapPointSet;
};