#pragma once
#include <memory>
#include <mutex>
#include <iostream>
#include "utils/numType.h"
#include "dataStruct/keyFrame.h"
#include "mapViewer/mapViewer.h"

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

private:
    std::mutex KFBufferMutex;
    std::deque<std::shared_ptr<KeyFrame>> KFBuffer;

    bool shouldQuit;

    bool hasNewKeyFrameToTest();
    void processNewKeyFrame();
    void checkLoopClosingCandidates();
    void optimizeEntireMap();

    std::shared_ptr<KeyFrame> KFToTest;
    bool needOptimize;
};