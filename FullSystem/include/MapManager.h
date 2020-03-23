#include "Map.h"

namespace SLAM
{

class MapManager
{
public:
    MapManager();
    void MakeNewMap();
    Map *GetActiveMap();
    std::vector<Map *> GetAllMaps();

private:
    long unsigned int mActiveId;
    std::map<long unsigned int, Map *> mpMaps;
};

}; // namespace SLAM