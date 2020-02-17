#pragma once
#include <memory>
#include <mutex>
#include <iostream>
#include "utils/numType.h"
#include "DataStruct/keyFrame.h"
#include "MapViewer/mapViewer.h"
#include "DataStruct/map.h"

class LoopCloser
{
public:
    LoopCloser();
    void testKeyFrame(std::shared_ptr<KeyFrame> KF);
    void run();

    inline void setShouldQuit()
    {
        shouldQuit = true;
    }

    inline void setMap(Map *map)
    {
        this->map = map;
    }

    inline void setMapViewer(MapViewer *viewer)
    {
        this->viewer = viewer;
    }

private:
    bool hasNewKeyFrameToTest();
    void processNewKeyFrame();
    void checkLoopClosingCandidates();
    void optimizeEntireMap();

    std::mutex KFBufferMutex;
    std::deque<std::shared_ptr<KeyFrame>> KFBuffer;

    std::shared_ptr<KeyFrame> KFToTest;
    bool needOptimize;

    Map *map;
    MapViewer *viewer;
    bool shouldQuit;
};