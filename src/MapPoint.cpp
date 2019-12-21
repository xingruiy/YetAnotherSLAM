#include "MapPoint.h"
#include <cmath>

// size_t MapPoint::nextMpId = 0;

// MapPoint::MapPoint()
//     : mpId(nextMpId++),
//       setToRemove(false),
//       referenceCounter(1)
// {
// }

//////////////////////////////////////////////

MapPoint::MapPoint(const Eigen::Vector3d &pos, KeyFrame *pRefKF, Map *pMap)
    : mpMap(pMap), mpRefKF(pRefKF), mWorldPos(pos)
{
}

std::map<KeyFrame *, size_t> MapPoint::GetObservations()
{
}

int MapPoint::Observations()
{
}

void MapPoint::AddObservation(KeyFrame *pKF, size_t idx)
{
}

void MapPoint::EraseObservation(KeyFrame *pKF)
{
}

void MapPoint::SetBadFlag()
{
}

bool MapPoint::isBad()
{
}

void MapPoint::Replace(MapPoint *pMP)
{
}

MapPoint *MapPoint::GetReplaced()
{
}