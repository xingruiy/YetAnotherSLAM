#include "mapPoint.h"
#include <cmath>

size_t MapPoint::nextMpId = 0;

MapPoint::MapPoint()
    : mpId(nextMpId++),
      setToRemove(false),
      referenceCounter(1)
{
}
