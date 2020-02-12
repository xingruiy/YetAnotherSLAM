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
    void CreateNewMapPoints();
    void UpdateConnections();

    // keyframe candidate
    std::mutex frameMutex;
    std::list<Frame *> newFrameQueue;
    Frame *currentFrame;
    KeyFrame *currentKeyFrame;
    KeyFrame *referenceKeyframe;

    ORB_SLAM2::ORBextractor *ORBExtractor;
    std::vector<KeyFrame *> localKeyFrames;
    std::vector<MapPoint *> localMapPoints;

    Map *mpMap;
};

} // namespace SLAM