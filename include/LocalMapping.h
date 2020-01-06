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

    void MatchLocalPoints();
    void CreateNewMapPoints();
    void SetShouldQuit();
    void UpdateLocalMap();
    void TrackLocalMap();

private:
    Map *mpMap;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;
    bool mbShouldQuit;

    KeyFrame *mpCurrentKeyFrame;

    std::list<KeyFrame *> mlNewKeyFrames;
    std::list<MapPoint *> mlpRecentAddedMapPoints;

    //Local Map
    KeyFrame *mpReferenceKF;
    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;
};