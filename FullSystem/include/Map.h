#pragma once
#include <set>
#include <mutex>
#include <memory>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "VoxelMap.h"

namespace SLAM
{

class Frame;
class MapPoint;
class KeyFrame;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame *pKF);
    void AddMapPoint(MapPoint *pMP);
    void AddMapStruct(MapStruct *pMS);

    void EraseKeyFrame(KeyFrame *pKF);
    void EraseMapPoint(MapPoint *pMP);
    void EraseMapStruct(MapStruct *pMS);

    void SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs);

    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame *> GetAllKeyFrames();
    std::vector<MapPoint *> GetAllMapPoints();
    std::vector<MapStruct *> GetAllVoxelMaps();

    std::vector<MapPoint *> GetReferenceMapPoints();

    long unsigned int MapPointsInMap();
    long unsigned KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void reset();

    std::vector<KeyFrame *> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

protected:
    struct cmp
    {
        bool operator()(MapStruct *a, MapStruct *b) const
        {
            return a->mnId < b->mnId;
        };
    };

    std::set<KeyFrame *> mspKeyFrames;
    std::set<MapPoint *> mspMapPoints;

    std::set<MapStruct *, cmp> mspMapStructs;

    std::vector<MapPoint *> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;

    std::mutex mFractualMutex;
};

} // namespace SLAM