#include "Map.h"
#include <fstream>
#include <algorithm>

namespace slam
{

using namespace std;

Map::Map() : mnBigChangeIdx(0),
             mnMaxKFid(0),
             mpMapStructOrigin(nullptr)
{
}

vector<MapPoint *> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
}

vector<KeyFrame *> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
}

vector<MapStruct *> Map::GetAllVoxelMaps()
{
    unique_lock<mutex> lock(mFractualMutex);
    return vector<MapStruct *>(mspMapStructs.begin(), mspMapStructs.end());
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);

    if (pKF->mnId > mnMaxKFid)
        mnMaxKFid = pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::AddMapStruct(MapStruct *pMS)
{
    unique_lock<mutex> lock(mFractualMutex);
    mspMapStructs.insert(pMS);

    if (!mpMapStructOrigin)
        mpMapStructOrigin = pMS;
}

void Map::reset()
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.clear();
    mspMapPoints.clear();
    mspMapStructs.clear();
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
}

void Map::EraseMapStruct(MapStruct *pMS)
{
    unique_lock<mutex> lock(mFractualMutex);
    mspMapStructs.erase(pMS);
}

void Map::setRefPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint *> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

} // namespace slam