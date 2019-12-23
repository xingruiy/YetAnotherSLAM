#include "Map.h"
#include <fstream>

vector<MapPoint *> Map::GetMapPointVec()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

unsigned long Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}