#pragma once
#include <memory>
#include <mutex>
#include <ORBextractor.h>
#include <ORBVocabulary.h>

#include "Map.h"
#include "GlobalDef.h"
#include "KeyFrame.h"
#include "Viewer.h"
#include "LoopClosing.h"

namespace SLAM
{

class Viewer;
class LoopClosing;

class LocalMapping
{
public:
    LocalMapping(ORB_SLAM2::ORBVocabulary *pVoc, Map *pMap);
    void AddKeyFrameCandidate(KeyFrame *pKF);
    void setLoopCloser(LoopClosing *pLoopCloser);
    void setViewer(Viewer *pViewer);
    void reset();
    void Run();

private:
    void ProcessNewKeyFrame();
    bool HasFrameToProcess();
    int MatchLocalPoints();
    void KeyFrameCulling();
    void SearchInNeighbors();
    void CreateNewMapPoints();
    void UpdateLocalMap();
    void UpdateKeyFrame();
    void MapPointCulling();

    // keyframe candidate
    std::mutex mMutexKeyFrameQueue;
    std::list<KeyFrame *> mlpKeyFrameQueue;
    KeyFrame *mpCurrentKeyFrame;
    KeyFrame *mpLastKeyFrame;
    KeyFrame *mpReferenceKeyframe;

    ORB_SLAM2::ORBVocabulary *ORBvocabulary;

    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;

    Map *mpMap;
    Viewer *mpViewer;
    LoopClosing *mpLoopCloser;
    std::list<MapPoint *> mlpRecentAddedMapPoints;

    cv::Mat mImg;
};

} // namespace SLAM