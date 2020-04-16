#include "MapManager.h"

namespace slam
{

MapManager::MapManager()
{
    pMap = new Map();
}

void MapManager::MakeNewMap(Map *pMap)
{
    // mActiveId = pMap->GetMapId();
    // mpMaps[mActiveId] = pMap;
}

void MapManager::Reset()
{
    // for (auto mit : mpMaps)
    //     mit.second->reset();
    // mpMaps.clear();
    pMap->reset();
    // Map::nextId = 0;
    // mActiveId = 0;
}

int MapManager::MapsInSystem()
{
    return mpMaps.size();
}

Map *MapManager::GetActiveMap()
{
    // return mpMaps[mActiveId]
    return pMap;
}

Map *MapManager::GetMap(long unsigned int id)
{
    return mpMaps[id];
}

void MapManager::FuseMap(long unsigned int id, long unsigned int id2)
{
    Map *pMap = mpMaps[id];
    Map *pMap2 = mpMaps[id2];
    if (pMap == nullptr || pMap2 == nullptr)
        return;

    pMap->FuseMap(pMap2);
    mpMaps[id2] == nullptr;
    mActiveId = id;
}

std::vector<Map *> MapManager::GetAllMaps()
{
    std::vector<Map *> allMaps;
    for (auto mit : mpMaps)
    {
        allMaps.push_back(mit.second);
    }
    return allMaps;
}

}; // namespace slam