#pragma once
#include "utils/numType.h"
#include "utils/frame.h"
#include <mutex>
#include <memory>

class Frame;

class MapPoint
{
    size_t id;
    static size_t nextId;

    // position in global coordinate
    Vec3d pos;
    Vec3d normal;

    Mat bestDesc;
    bool invalidateFlag;
    std::mutex mutexPos, mutexDesc;
    std::shared_ptr<Frame> referenceKF;
    std::map<std::shared_ptr<Frame>, Vec3d> observations;

public:
    MapPoint(const Vec3d &pt, std::shared_ptr<Frame> refKF);
    void addObservation(std::shared_ptr<Frame> frame, const Vec3d &obs);
    void removeObservation(std::shared_ptr<Frame> frame);
    void invalidatePoint();
    bool isValid() const;
    size_t getNumObservations() const;
    std::shared_ptr<Frame> getReferenceKF() const;
};