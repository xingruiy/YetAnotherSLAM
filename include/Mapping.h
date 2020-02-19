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

class Mapping
{
public:
    Mapping(ORB_SLAM2::ORBVocabulary *pVoc, Map *map, Viewer *pViewer);
    void AddKeyFrameCandidate(const Frame &F);
    void setLoopCloser(LoopClosing *pLoopCloser);
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
    Viewer *mpViewer;
    LoopClosing *mpLoopCloser;
    std::list<MapPoint *> mlpRecentAddedMapPoints;

    cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
};

} // namespace SLAM