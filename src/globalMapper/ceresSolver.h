#pragma once
#include "globalMapper/sophusEigenHack.h"
#include "utils/numType.h"
#include <unordered_map>

class CeresSolver
{
    double K[4];
    // std::shared_ptr<ceres::Problem> solver;
    std::unordered_map<size_t, Vec3d> mapPoints;
    std::unordered_map<size_t, SE3> cameras;
    std::unordered_map<size_t, std::unordered_map<size_t, Vec2d>> observations;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CeresSolver(Mat33d &K);

    void reset();
    bool hasCamera(const size_t camIdx);
    SE3 getCamera(const size_t camIdx);
    void optimize(const int maxiter, const size_t oldestKFId, const size_t newestKFId);
    void addCamera(const size_t camIdx, const SE3 &T, bool fixed);
    void removeCamera(const size_t camIdx);
    void addWorldPoint(const size_t ptIdx, const Vec3d &pos, bool fixed);
    void removeWorldPoint(const size_t ptIdx);
    void addObservation(const size_t ptIdx, const size_t camIdx, const Vec2d &obs);
    void removeObservation(const size_t ptIdx, const size_t camIdx);
};