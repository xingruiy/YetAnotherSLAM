#include "mapPoint.h"

size_t MapPoint::nextId = 0;

MapPoint::MapPoint()
    : id(nextId++)
{
}

MapPoint::MapPoint(std::shared_ptr<Frame> hostKF, const Vec3d &posWorld, Mat desc)
    : id(nextId++), hostKF(hostKF), position(posWorld), descriptor(desc)
{
}

size_t MapPoint::getId() const
{
  return id;
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

std::unordered_map<std::shared_ptr<Frame>, Vec3d> MapPoint::getObservations() const
{
  return observations;
}

void MapPoint::fusePoint(std::shared_ptr<MapPoint> &other)
{
  auto obs = other->getObservations();
  observations.insert(obs.begin(), obs.end());
  other = NULL;
}

size_t MapPoint::getNumObservations() const
{
  return observations.size();
}

Vec3d MapPoint::getPosWorld() const
{
  return position;
}

std::shared_ptr<Frame> MapPoint::getHost() const
{
  return hostKF;
}

void MapPoint::setHost(std::shared_ptr<Frame> frame)
{
  hostKF = frame;
}

void MapPoint::setPosWorld(const Vec3d &pos)
{
  position = pos;
}

void MapPoint::setDescriptor(const Mat &desc)
{
  descriptor = desc;
}

Mat MapPoint::getDescriptor() const
{
  return descriptor;
}