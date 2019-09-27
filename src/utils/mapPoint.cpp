#include "mapPoint.h"

size_t MapPoint::nextId = 0;

MapPoint::MapPoint()
    : id(nextId++),
      inOptimizer(false),
      invalidated(false),
      numObservations(1),
      isImmature(true)
{
}

double *MapPoint::getParameterBlock()
{
  return position.data();
}

void MapPoint::removeObservation(std::shared_ptr<Frame> kf)
{
  auto result = observations.find(kf);
  if (result != observations.end())
    observations.erase(result);
}

void MapPoint::addObservation(std::shared_ptr<Frame> kf, const Vec3d &obs)
{
  observations.insert(std::make_pair(kf, obs));
}

size_t MapPoint::getNumObservations() const
{
  return observations.size();
}