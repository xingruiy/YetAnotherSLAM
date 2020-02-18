#pragma once
#include "Map.h"
#include "Tracking.h"
#include "Mapping.h"

namespace SLAM
{

class Map;
class Tracking;
class Mapping;

class LoopFinder
{
public:
    LoopFinder();
    void Run();

    typedef std::pair<std::set<KeyFrame *>, int> ConsistentGroup;

private:
    bool CheckNewKeyFrames();
    bool DetectLoop();

    Map *mpMap;
    Tracking *mpTracker;
    Mapping *mpLocalMapper;

    std::mutex mMutexLoopQueue;
    std::list<KeyFrame *> mlpLoopKeyFrameQueue;

    KeyFrame *mpCurrentKF;
    KeyFrame *mpMatchedKF;

    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFrame *> mvpEnoughConsistentCandidates;
    std::vector<KeyFrame *> mvpCurrentConnectedKFs;
    std::vector<MapPoint *> mvpCurrentMatchedPoints;
    std::vector<MapPoint *> mvpLoopMapPoints;
};

} // namespace SLAM
