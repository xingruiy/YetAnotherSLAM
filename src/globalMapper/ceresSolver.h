#pragma once
#include "utils/numType.h"
#include "globalMapper/sophusEigenHack.h"
#include "globalMapper/ceresStructure.h"
#include <unordered_map>

using PointBlockMap = std::unordered_map<size_t, std::shared_ptr<PointBlock>>;
using CameraBlockMap = std::unordered_map<size_t, std::shared_ptr<CameraBlock>>;
using ResidualBlockMap = std::unordered_map<size_t, std::unordered_map<size_t, Vec2d>>;

class CeresSolver
{
    double K[4];
    PointBlockMap ptBlockMap;
    CameraBlockMap camBlockMap;
    ResidualBlockMap resBlockMap;

    std::shared_ptr<ceres::Problem> solver;
    ceres::Solver::Options solverOptions;
    ceres::Problem::Options problemOptions;
    ceres::Solver::Summary solverLog;

    std::shared_ptr<PointBlock> getPointBlock(const size_t ptId) const;
    std::shared_ptr<CameraBlock> getCameraBlock(const size_t camId) const;
    void initializeParameters();
    bool hasResidualBlock(const size_t ptId, const size_t camId) const;
    void addResidualBlock(
        // std::shared_ptr<ceres::CostFunction> costFunction,
        // std::shared_ptr<ceres::LossFunction> lossFunction,
        std::shared_ptr<CameraBlock> cameraBlock,
        std::shared_ptr<PointBlock> pointBlock,
        ResidualBlock &resBlock);
    void removeResidualBlock(ceres::ResidualBlockId id);
    void removeResidualBlock(ResidualBlock &block);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CeresSolver(Mat33d &K);
    ~CeresSolver();

    void reset();

    bool hasPoint(const size_t ptIdx) const;
    bool hasCamera(const size_t camIdx) const;

    Vec3d getPtPosOptimized(const size_t ptId) const;
    SE3 getCamPoseOptimized(const size_t camId) const;
    void setCameraBlockConstant(const size_t camId);
    void optimize(const int maxiter);
    void addCamera(const size_t camIdx, const SE3 &T, bool fixed);
    void removeCamera(const size_t camIdx);
    void addWorldPoint(const size_t ptIdx, const Vec3d &pos, bool fixed);
    void removeWorldPoint(const size_t ptIdx);
    void addObservation(const size_t ptIdx, const size_t camIdx, const Vec2d &obs);
    void removeObservation(const size_t ptIdx, const size_t camIdx);

    std::vector<Vec3d> getActivePoints() const;
    Mat33d getCamParamOptimized() const;
};