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
    void reset();

    void AddKeyFrame(KeyFrame *pKF);
    void EraseKeyFrame(KeyFrame *pKF);

    void AddMapPoint(MapPoint *pMP);
    void EraseMapPoint(MapPoint *pMP);

    void AddMapStruct(MapStruct *pMS);
    void EraseMapStruct(MapStruct *pMS);

    void SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs);

    std::mutex mMutexMapUpdate;

    std::vector<MapPoint *> GetReferenceMapPoints();
    std::vector<KeyFrame *> GetAllKeyFrames();
    std::vector<MapPoint *> GetAllMapPoints();
    std::vector<MapStruct *> GetAllVoxelMaps();

private:
    std::mutex mMapMutex;
    std::mutex mFractualMutex;
    std::set<KeyFrame *> mspKeyFrames;
    std::set<MapPoint *> mspMapPoints;
    std::set<MapStruct *> mspMapStructs;
    std::vector<MapPoint *> mvpReferenceMapPoints;
};

} // namespace SLAM