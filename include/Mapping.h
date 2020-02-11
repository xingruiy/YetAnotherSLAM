#pragma once
#include <memory>
#include <mutex>
#include <ORBextractor.h>
#include "Map.h"
#include "GlobalDef.h"
#include "KeyFrame.h"
#include "Viewer.h"

namespace SLAM
{

class Viewer;
class MapViewer;

class Mapping
{
public:
    Mapping(Map *map);
    void addKeyFrameCandidate(Frame *F);
    void reset();
    void Run();

private:
    void MakeNewKeyFrame();
    bool HasFrameToProcess();
    void LookforPointMatches();
    void KeyFrameCulling();
    void SearchInNeighbors();

    ORB_SLAM2::ORBextractor *ORBExtractor;
    std::vector<KeyFrame *> localKeyFrames;
    std::vector<MapPoint *> localMapPoints;

public:
    void InsertKeyFrame(KeyFrame *pKF);

protected:
    // void doTasks();

    // void MatchLocalPoints();
    // void CreateNewMapPoints();
    // void UpdateLocalMap();

    // The global map
    Map *mpMap;

    std::mutex frameMutex;
    std::list<Frame *> newFrameQueue;
    Frame *currentFrame;
    KeyFrame *currentKeyFrame;

    // This is to store new keyframes which are to be processed
    std::mutex mMutexNewKFs;
    std::list<KeyFrame *> mlNewKeyFrames;
    KeyFrame *mpCurrentKeyFrame;

    // Local Map, highly volatile
    KeyFrame *mpReferenceKF;
    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;

private:
    std::vector<KeyFrame *> keyframes;
    std::vector<MapPoint *> mapStruct;
};

} // namespace SLAM