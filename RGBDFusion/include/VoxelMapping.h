#ifndef VOXEL_MAPPING
#define VOXEL_MAPPING

#include "VoxelMap.h"
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

class VoxelMapping
{
  Eigen::Matrix3d mK;
  MapStruct deviceMap;

  // for map udate
  uint mNumVisibleBlocks;

  // for raycast
  cv::cuda::GpuMat zRangeX;
  cv::cuda::GpuMat zRangeY;
  uint mNumRenderingBlocks;
  RenderingBlock *mplRenderingBlock;

public:
  ~VoxelMapping();
  VoxelMapping(const int w, const int h, const Eigen::Matrix3f &K);

  void FuseFrame(cv::cuda::GpuMat depth, const Sophus::SE3d &T);
  void TracingDepth(cv::cuda::GpuMat &vertex, const Sophus::SE3d &T);

  void reset();

  size_t fetchMeshWithNormal(void *vertex, void *normal);
};

#endif