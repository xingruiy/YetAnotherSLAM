#include "MapManager.h"

namespace SLAM
{

MapManager::MapManager()
{
    MakeNewMap();
}

void MapManager::MakeNewMap()
{
    Map *pMap = new Map();
    mActiveId = pMap->GetMapId();
    mpMaps[mActiveId] = pMap;
}

Map *MapManager::GetActiveMap()
{
    return mpMaps[mActiveId];
}

std::vector<Map *> MapManager::GetAllMaps()
{
}

}; // namespace SLAM