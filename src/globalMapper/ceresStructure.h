#pragma once
#include <ceres/ceres.h>

struct ResidualBlock
{
  ceres::ResidualBlockId residualBlockId;
  ceres::CostFunction *costFunction;
  ceres::LossFunction *lossFunction;
};

struct CameraBlock
{
  size_t cameraId;
  SE3 lastSuccessOptimized;
  SE3 optimizationBuffer;

  inline CameraBlock(
      const size_t cameraId,
      const SE3 &initTransform)
      : cameraId(cameraId),
        optimizationBuffer(initTransform),
        lastSuccessOptimized(initTransform)
  {
  }
};

struct PointBlock
{
  size_t pointId;
  Vec3d lastSuccessOptimized;
  Vec3d optimizationBuffer;
  bool potentialOutlier;

  inline PointBlock(
      const size_t pointId,
      const Vec3d &initPos)
      : pointId(pointId),
        optimizationBuffer(initPos),
        lastSuccessOptimized(initPos),
        potentialOutlier(false)
  {
  }
};