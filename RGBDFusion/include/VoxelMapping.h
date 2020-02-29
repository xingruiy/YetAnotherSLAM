#ifndef VOXEL_MAPPING
#define VOXEL_MAPPING

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#include "VoxelMap.h"
#include "MeshEngine.h"

class VoxelMapping
{

public:
  ~VoxelMapping();
  VoxelMapping(const int w, const int h, const Eigen::Matrix3f &K);

  void FuseFrame(cv::cuda::GpuMat depth, const Sophus::SE3d &T);
  void TracingDepth(cv::cuda::GpuMat &vertex, const Sophus::SE3d &T);

  void reset();

  void TryMeshification();
  MapStruct *GetMapStruct();
  // size_t fetchMeshWithNormal(void *vertex, void *normal);

private:
  Eigen::Matrix3d mK;
  MapStruct deviceMap;

  // for map udate
  uint mNumVisibleBlocks;

  // for raycast
  cv::cuda::GpuMat zRangeX;
  cv::cuda::GpuMat zRangeY;
  uint mNumRenderingBlocks;
  RenderingBlock *mplRenderingBlock;

  MeshEngine *mpMeshEngine;
};

#endif