#include "Map.h"

namespace SLAM
{

std::vector<MapPoint *> Map::GetAllMapPoints()
{
    std::unique_lock<std::mutex> lock(mMapMutex);
    return std::vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
}

std::vector<KeyFrame *> Map::GetAllKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMapMutex);
    return std::vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
}

std::vector<MapStruct *> Map::GetAllVoxelMaps()
{
    std::unique_lock<std::mutex> lock(mFractualMutex);
    return std::vector<MapStruct *>(mspMapStructs.begin(), mspMapStructs.end());
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMapMutex);
    mspKeyFrames.insert(pKF);
}

void Map::AddMapPoint(MapPoint *pMP)
{
    std::unique_lock<std::mutex> lock(mMapMutex);
    mspMapPoints.insert(pMP);
}

void Map::reset()
{
    std::unique_lock<std::mutex> lock(mMapMutex);
    mspKeyFrames.clear();
    mspMapPoints.clear();
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    std::unique_lock<std::mutex> lock(mMapMutex);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMapMutex);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the KeyFrame
}

void Map::AddMapStruct(MapStruct *pMS)
{
    std::unique_lock<std::mutex> lock(mFractualMutex);
    mspMapStructs.insert(pMS);
}

void Map::EraseMapStruct(MapStruct *pMS)
{
    std::unique_lock<std::mutex> lock(mFractualMutex);
    mspMapStructs.erase(pMS);
}

void Map::SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs)
{
    std::unique_lock<std::mutex> lock(mMapMutex);
    mvpReferenceMapPoints = vpMPs;
}

std::vector<MapPoint *> Map::GetReferenceMapPoints()
{
    std::unique_lock<std::mutex> lock(mMapMutex);
    return mvpReferenceMapPoints;
}

} // namespace SLAM