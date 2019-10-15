#include "mapPoint.h"
#include <cmath>

size_t MapPoint::nextId = 0;

MapPoint::MapPoint()
    : id(nextId++), bad(false), mature(false)
{
}

MapPoint::MapPoint(
    std::shared_ptr<Frame> hostKF,
    const Vec3d &posWorld,
    Mat desc)
    : id(nextId++),
      hostKF(hostKF),
      position(posWorld),
      descriptor(desc),
      bad(false),
      mature(false)
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
  if (other)
    std::unique_lock<std::mutex> lock(other->lock);
  auto obs = other->getObservations();
  observations.insert(obs.begin(), obs.end());
  position = (position + other->position) / 2.0;
  other->flagBad();
  other = NULL;
}

size_t MapPoint::getNumObservations() const
{
  return observations.size();
}

Vec3d MapPoint::getPosWorld() const
{
  if (isMature())
    return position;
  else
    return hostKF->getPoseInGlobalMap() * position;
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

bool MapPoint::isBad() const
{
  return bad;
}

void MapPoint::flagBad()
{
  bad = true;
}

bool MapPoint::isMature() const
{
  return mature && observations.size() > 1;
}

void MapPoint::setMature()
{
  mature = true;
  position = hostKF->getPoseInGlobalMap() * position;
}

bool MapPoint::checkParallaxAngle() const
{
  if (isMature())
    return false;

  auto hostPos = hostKF->getPositionWorld();
  Vec3d hostDir = hostPos - getPosWorld();
  for (auto obs : observations)
  {
    auto &kf = obs.first;
    if (kf == hostKF)
      continue;

    auto pos = kf->getPositionWorld();
    Vec3d dir = pos - getPosWorld();
    double cosa = hostDir.dot(dir) / (hostDir.norm() * dir.norm());
    double angle = std::acos(cosa);
    if (angle > 0.3)
      return true;
  }
}