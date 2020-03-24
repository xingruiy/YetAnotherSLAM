#include "MapManager.h"

namespace SLAM
{

MapManager::MapManager()
{
}

void MapManager::MakeNewMap(Map *pMap)
{
    mActiveId = pMap->GetMapId();
    mpMaps[mActiveId] = pMap;
}

Map *MapManager::GetActiveMap()
{
    return mpMaps[mActiveId];
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
}

}; // namespace SLAM