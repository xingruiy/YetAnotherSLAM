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

    void WriteToFile(const std::string &strFile);
    void ReadFromFile(const std::string &strFile);

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

    std::set<MapStruct *> mspMapStructs;

    std::vector<MapPoint *> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;

    std::mutex mFractualMutex;

public:
    long unsigned int GetMapId();

protected:
    long unsigned int mMapId;
    static long unsigned int nextId;
};


} // namespace SLAM