#pragma once
#include <memory>
#include <mutex>
#include "Map.h"

#include "KeyFrame.h"
#include "Viewer.h"

class MapViewer;

class LocalMapping
{
public:
    LocalMapping(Map *pMap);

    void Spin();

    void InsertKeyFrame(KeyFrame *pKF);

    bool CheckNewKeyFrames();

    void ProcessNewKeyFrame();

    void MapPointCulling();
    void CreateNewMapPoints();

private:
    Map *mpMap;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    KeyFrame *mpCurrentKeyFrame;

    std::list<KeyFrame *> mlNewKeyFrames;
    std::list<MapPoint *> mlpRecentAddedMapPoints;
};