#include "mapPoint.h"

size_t MapPoint::nextId = 0;

MapPoint::MapPoint(const Vec3d &pt, std::shared_ptr<Frame> refKF)
    : pos(pt), referenceKF(refKF), invalidateFlag(false)
{
    id = nextId++;
}
