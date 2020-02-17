#include "DataStruct/map.h"
#include <fstream>

Map::Map()
{
}

void Map::clear()
{
    std::unique_lock<std::mutex> lock(mapMutex);
    descriptorDB.release();
    keyFrameDB.clear();
    mapPointDB.clear();
}

void Map::addKeyFrame(std::shared_ptr<KeyFrame> KF)
{
    std::unique_lock<std::mutex> lock(mapMutex);
    keyFrameDB.push_back(KF);
}

void Map::addMapPoint(std::shared_ptr<MapPoint> pt)
{
    std::unique_lock<std::mutex> lock(mapMutex);
    descriptorDB.push_back(pt->descriptor);
    mapPointDB.push_back(pt);
}

void Map::getMapPoint(std::vector<Vec3f> &MPs)
{
    MPs.clear();
    {
        std::unique_lock<std::mutex> lock(mapMutex);
        for (auto &mp : mapPointDB)
        {
            MPs.push_back(mp->pos.cast<float>());
        }
    }
}

void Map::writeToDisk(const char *fileName)
{
    std::ofstream file(fileName);
    if (file.is_open())
    {
    }

    file.close();
}

void Map::readFromDisk(const char *fileName)
{
    std::ifstream file(fileName);
    if (file.is_open())
    {
    }

    file.close();
}