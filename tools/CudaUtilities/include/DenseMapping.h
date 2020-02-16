#pragma once

#include "VoxelMap.h"
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

class DenseMapping
{
  Eigen::Matrix3d mK;
  MapStruct deviceMap;

  // for map udate
  uint count_visible_block;

  // for raycast
  cv::cuda::GpuMat zRangeX;
  cv::cuda::GpuMat zRangeY;
  cv::cuda::GpuMat BufferFloat4WxH;
  uint count_rendering_block;
  RenderingBlock *listRenderingBlock;

public:
  ~DenseMapping();
  DenseMapping(int w, int h, const Eigen::Matrix3d &K);

  void reset();
  void fuseFrame(cv::cuda::GpuMat depth, const Sophus::SE3d &T, unsigned int id);
  void raytrace(const Sophus::SE3d &T);
  size_t fetchMeshWithNormal(void *vertex, void *normal);
  cv::cuda::GpuMat GetSyntheticVertexMap();
};
