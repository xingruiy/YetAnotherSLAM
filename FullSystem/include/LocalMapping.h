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
    void AddKeyFrameCandidate(const Frame &F);
    void setLoopCloser(LoopClosing *pLoopCloser);
    void setViewer(Viewer *pViewer);
    void reset();
    void Run();

private:
    void CreateNewKeyFrame();
    bool HasFrameToProcess();
    int MatchLocalPoints();
    void KeyFrameCulling();
    void SearchInNeighbors();
    void CreateNewMapPoints();
    void UpdateLocalMap();
    void UpdateKeyFrame();
    void MapPointCulling();

    // keyframe candidate
    std::mutex frameMutex;
    Frame mCurrentFrame;
    std::list<Frame> mlFrameQueue;
    KeyFrame *NextKeyFrame;
    KeyFrame *mLastKeyFrame;
    KeyFrame *referenceKeyframe;

    ORBextractor *mpExtractor;
    ORB_SLAM2::ORBVocabulary *ORBvocabulary;

    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;

    Map *mpMap;
    Viewer *mpViewer;
    LoopClosing *mpLoopCloser;
    std::list<MapPoint *> mlpRecentAddedMapPoints;

    cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    cv::Mat mImg;
};

} // namespace SLAM