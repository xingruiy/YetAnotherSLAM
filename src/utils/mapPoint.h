#pragma once
#include "utils/numType.h"
#include <memory>
#include <unordered_map>

class Frame;

class MapPoint
{
    static size_t nextId;

public:
    MapPoint();
    double *getParameterBlock();
    size_t getNumObservations() const;
    void removeObservation(std::shared_ptr<Frame> kf);
    void addObservation(std::shared_ptr<Frame> kf, const Vec3d &obs);

    size_t id;
    int numObservations;
    bool visited;
    bool invalidated;
    bool inOptimizer;
    Vec3d position;
    Vec3d relativePos;
    // Vec9f descriptor;
    Mat descriptor;
    bool isImmature;
    std::shared_ptr<Frame> hostKF;
    std::unordered_map<std::shared_ptr<Frame>, Vec3d> observations;
};