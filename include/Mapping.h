#pragma once
#include <memory>
#include <mutex>
#include "Map.h"

#include "KeyFrame.h"
#include "Viewer.h"

namespace SLAM
{

class Viewer;
class MapViewer;

class Mapping
{
public:
    Mapping(Map *pMap, Viewer *pViewer);
    void InsertKeyFrame(KeyFrame *pKF);

    void Run();

protected:
    void doTasks();
    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();

    void MatchLocalPoints();
    void CreateNewMapPoints();
    void UpdateLocalMap();
    void KeyFrameCulling();
    void SearchInNeighbors();

    // The global map
    Map *mpMap;

    // Map Viewer
    Viewer *viewer;

    // This is to store new keyframes which are to be processed
    std::mutex mMutexNewKFs;
    std::list<KeyFrame *> mlNewKeyFrames;
    KeyFrame *mpCurrentKeyFrame;

    // Local Map, highly volatile
    KeyFrame *mpReferenceKF;
    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;

private:
    std::vector<FrameShell *> frames;
    std::vector<KeyFrame *> keyframes;
    std::vector<MapPoint *> mapStruct;
};

} // namespace SLAM