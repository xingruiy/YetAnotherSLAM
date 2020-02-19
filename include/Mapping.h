#pragma once
#include <memory>
#include <mutex>
#include <ORBextractor.h>
#include <ORBVocabulary.h>

#include "Map.h"
#include "GlobalDef.h"
#include "KeyFrame.h"
#include "Viewer.h"
#include "LoopFinder.h"

namespace SLAM
{

class LoopFinder;

class Mapping
{
public:
    Mapping(ORB_SLAM2::ORBVocabulary *pVoc, Map *map);
    void AddKeyFrameCandidate(const Frame &F);
    void setLoopCloser(LoopFinder *pLoopCloser);
    void reset();
    void Run();

private:
    void MakeNewKeyFrame();
    bool HasFrameToProcess();
    int MatchLocalPoints();
    void KeyFrameCulling();
    void SearchInNeighbors();
    void TriangulatePoints();
    void CreateNewMapPoints();
    void UpdateConnections();
    void UpdateKeyFrame();

    // keyframe candidate
    std::mutex frameMutex;
    Frame NextFrame;
    std::list<Frame> mlFrameQueue;
    KeyFrame *NextKeyFrame;
    KeyFrame *lastKeyFrame;
    KeyFrame *referenceKeyframe;

    ORB_SLAM2::ORBextractor *ORBExtractor;
    ORB_SLAM2::ORBVocabulary *ORBvocabulary;

    std::vector<KeyFrame *> localKeyFrames;
    std::vector<MapPoint *> localMapPoints;

    Map *mpMap;
    LoopFinder *mpLoopCloser;
    std::list<MapPoint *> mlpRecentAddedMapPoints;

    cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
};

} // namespace SLAM