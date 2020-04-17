#pragma once
#include <set>
#include <mutex>
#include <memory>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "VoxelMap.h"

namespace slam
{

class Frame;
class MapPoint;
class KeyFrame;

class Map
{
public:
    Map();
    void reset();

    void AddKeyFrame(KeyFrame *pKF);
    void AddMapPoint(MapPoint *pMP);
    void AddMapStruct(MapStruct *pMS);

    void EraseKeyFrame(KeyFrame *pKF);
    void EraseMapPoint(MapPoint *pMP);
    void EraseMapStruct(MapStruct *pMS);
    void setRefPoints(const std::vector<MapPoint *> &vpMPs);

    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame *> GetAllKeyFrames();
    std::vector<MapPoint *> GetAllMapPoints();
    std::vector<MapStruct *> GetAllVoxelMaps();
    std::vector<MapPoint *> GetReferenceMapPoints();

    int MapPointsInMap();
    int KeyFramesInMap();
    int GetMaxKFid();

    std::vector<KeyFrame *> mvpKeyFrameOrigins;
    MapStruct *mpMapStructOrigin;
    std::mutex mMutexMapUpdate;
    std::mutex mPointCreateMutex;

protected:
    std::set<KeyFrame *> mspKeyFrames;
    std::set<MapPoint *> mspMapPoints;
    std::set<MapStruct *> mspMapStructs;
    std::vector<MapPoint *> mvpReferenceMapPoints;

    int mnMaxKFid;
    int mnBigChangeIdx;
    std::mutex mMutexMap;
    std::mutex mFractualMutex;
};

} // namespace slam