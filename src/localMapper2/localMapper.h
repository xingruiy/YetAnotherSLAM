#pragma once
#include <memory>
#include "utils/numType.h"
#include "localMapper2/denseMap.h"

class DenseMapping
{
public:
  ~DenseMapping();
  DenseMapping(int w, int h, const Eigen::Matrix3d &K);

  size_t fetch_mesh_vertex_only(void *vertex);
  size_t fetch_mesh_with_normal(void *vertex, void *normal);
  size_t fetch_mesh_with_colour(void *vertex, void *normal);

  void fuseFrame(GMat depth, const SE3 &T);
  void raytrace(GMat &vertex, const SE3 &T);
  void reset();

private:
  Eigen::Matrix3d cam_params;
  MapStruct device_map;

  // for map udate
  cv::cuda::GpuMat flag;
  cv::cuda::GpuMat pos_array;
  uint count_visible_block;
  HashEntry *visible_blocks;

  // for raycast
  cv::cuda::GpuMat zrange_x;
  cv::cuda::GpuMat zrange_y;
  uint count_rendering_block;
  RenderingBlock *rendering_blocks;
};
