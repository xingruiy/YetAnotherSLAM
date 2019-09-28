#pragma once
#include "utils/numType.h"
#include "utils/frame.h"
#include <memory>
#include <unordered_map>

class Frame;

class MapPoint
{
    size_t id;
    Vec3d position;
    Mat descriptor;
    std::shared_ptr<Frame> hostKF;
    static size_t nextId;
    std::unordered_map<std::shared_ptr<Frame>, Vec3d> observations;

public:
    MapPoint();
    MapPoint(std::shared_ptr<Frame> hostKF, const Vec3d &posWorld, Mat desc);

    Vec3d getPosWorld() const;
    size_t getId() const;
    std::shared_ptr<Frame> getHost() const;
    void setHost(std::shared_ptr<Frame> frame);
    void setPosWorld(const Vec3d &pos);
    void setDescriptor(const Mat &desc);

    double *getParameterBlock();
    Mat getDescriptor() const;

    void fusePoint(std::shared_ptr<MapPoint> &other);
    size_t getNumObservations() const;
    void removeObservation(std::shared_ptr<Frame> kf);
    void addObservation(std::shared_ptr<Frame> kf, const Vec3d &obs);
    std::unordered_map<std::shared_ptr<Frame>, Vec3d> getObservations() const;

    bool visited;
};