#include "Map.h"
#include <fstream>

namespace SLAM
{

Map::Map() : mnBigChangeIdx(0)
{
}

std::vector<MapPoint *> Map::GetAllMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return std::vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
}

std::vector<KeyFrame *> Map::GetAllKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return std::vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
}

std::vector<MapStruct *> Map::GetAllVoxelMaps()
{
    std::unique_lock<std::mutex> lock(mFractualMutex);
    return std::vector<MapStruct *>(mspMapStructs.begin(), mspMapStructs.end());
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);

    if (pKF->mnId > mnMaxKFid)
        mnMaxKFid = pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::AddMapStruct(MapStruct *pMS)
{
    std::unique_lock<std::mutex> lock(mFractualMutex);
    mspMapStructs.insert(pMS);
}

void Map::reset()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspKeyFrames.clear();
    mspMapPoints.clear();
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the KeyFrame
}

void Map::EraseMapStruct(MapStruct *pMS)
{
    std::unique_lock<std::mutex> lock(mFractualMutex);
    mspMapStructs.erase(pMS);
}

void Map::SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

long unsigned int Map::MapPointsInMap()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

std::vector<MapPoint *> Map::GetReferenceMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::WriteToFile(const std::string &strFile)
{
    std::ofstream file(strFile, std::ios_base::binary);
    if (file.is_open())
    {
        file << mspKeyFrames.size()
             << mspMapPoints.size();

        for (auto sit = mspKeyFrames.begin(), send = mspKeyFrames.end(); sit != send; ++sit)
        {
            KeyFrame *pKF = *sit;
            if (pKF == nullptr)
            {
                std::cout << "error writing files: keyframe shouldn't be nullptr..." << std::endl;
                continue;
            }

            auto vpMapPoints = pKF->GetMapPointMatches();
            for (auto vit = vpMapPoints.begin(), vend = vpMapPoints.end(); vit != vend; ++vit)
            {
                MapPoint *pMP = *vit;
                if (!pMP || pMP->isBad())
                {
                    file << -1;
                }
                else
                {
                    file << pMP->mnId;
                }
            }
        }
    }
    else
    {
        std::cout << "failed at creating files..." << std::endl;
        return;
    }
}

void Map::ReadFromFile(const std::string &strFile)
{
}

} // namespace SLAM