#include "Map.h"

namespace SLAM
{

class MapManager
{
public:
    MapManager();
    void Reset();
    int MapsInSystem();
    void MakeNewMap(Map *pMap);
    Map *GetActiveMap();
    Map *GetMap(long unsigned int id);
    void FuseMap(long unsigned int id, long unsigned int id2);
    std::vector<Map *> GetAllMaps();

private:
    long unsigned int mActiveId;
    std::map<long unsigned int, Map *> mpMaps;
};

}; // namespace SLAM