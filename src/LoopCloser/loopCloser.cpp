#include "LoopCloser/loopCloser.h"
#include <unistd.h>

LoopCloser::LoopCloser()
    : needOptimize(false), shouldQuit(false)
{
}

void LoopCloser::testKeyFrame(std::shared_ptr<KeyFrame> KF)
{
    std::unique_lock<std::mutex>(KFBufferMutex);
    KFBuffer.push_back(KF);
}

void LoopCloser::run()
{
    while (!shouldQuit)
    {
        if (hasNewKeyFrameToTest())
        {
            processNewKeyFrame();

            checkLoopClosingCandidates();

            optimizeEntireMap();

            usleep(3000);
        }
        else
        {
            usleep(3000);
        }
    }
}

bool LoopCloser::hasNewKeyFrameToTest()
{
    std::unique_lock<std::mutex> lock(KFBufferMutex);
    return !KFBuffer.empty();
}

void LoopCloser::processNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(KFBufferMutex);
        KFToTest = KFBuffer.front();
        KFBuffer.pop_front();
    }

    auto nPoints = KFToTest->keyPoints.size();
    if (nPoints == 0)
        return;
}

void LoopCloser::checkLoopClosingCandidates()
{
    auto &mapPoints = map->mapPointDB;
    auto &mapDesc = map->descriptorDB;
}

void LoopCloser::optimizeEntireMap()
{
}