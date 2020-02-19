#pragma once
#include "Map.h"
#include "Tracking.h"
#include "Mapping.h"
#include "KeyFrameDatabase.h"

namespace SLAM
{

class Map;
class Tracking;
class Mapping;

class LoopFinder
{
public:
    typedef std::pair<std::set<KeyFrame *>, int> ConsistentGroup;

    LoopFinder(Map *pMap, KeyFrameDatabase *pDB, ORB_SLAM2::ORBVocabulary *pVoc);
    void Run();
    void InsertKeyFrame(KeyFrame *pKF);

private:
    bool CheckNewKeyFrames();
    bool DetectLoop();
    void ComputeSim3();
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    Map *mpMap;
    Tracking *mpTracker;
    Mapping *mpLocalMapper;

    std::mutex mMutexLoopQueue;
    std::list<KeyFrame *> mlpLoopKeyFrameQueue;

    KeyFrameDatabase *mpKeyFrameDB;
    ORB_SLAM2::ORBVocabulary *mpORBVocabulary;

    KeyFrame *mpCurrentKF;
    KeyFrame *mpMatchedKF;

    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFrame *> mvpEnoughConsistentCandidates;
    std::vector<KeyFrame *> mvpCurrentConnectedKFs;
    std::vector<MapPoint *> mvpCurrentMatchedPoints;
    std::vector<MapPoint *> mvpLoopMapPoints;

    long unsigned int mLastLoopKFid;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    bool mbFinishedGBA;
    bool mbRunningGBA;
};

} // namespace SLAM
