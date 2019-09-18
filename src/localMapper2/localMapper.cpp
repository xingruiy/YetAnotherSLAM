#include <cuda_runtime_api.h>
#include "map_proc.h"
#include "localMapper.h"

DenseMapping::DenseMapping(int w, int h, const Eigen::Matrix3d &K) : cam_params(K)
{
  // device_map.create();#
  device_map.create(100000, 80000, 120000, 0.005f, 0.02f);
  device_map.reset();
  zrange_x.create(h / 8, w / 8, CV_32FC1);
  zrange_y.create(h / 8, w / 8, CV_32FC1);

  cudaMalloc((void **)&visible_blocks, sizeof(HashEntry) * 100000);
  cudaMalloc((void **)&rendering_blocks, sizeof(RenderingBlock) * 100000);
}

DenseMapping::~DenseMapping()
{
  device_map.release();
  cudaFree((void **)&visible_blocks);
  cudaFree((void **)&rendering_blocks);
}

void DenseMapping::fuseFrame(GMat depth, const SE3 &T)
{
  count_visible_block = 0;

  ::update(
      device_map,
      // device_map.state,
      depth,
      depth,
      T,
      cam_params,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::raytrace(GMat &vertex, const SE3 &T)
{
  if (count_visible_block == 0)
    return;

  ::create_rendering_blocks(
      device_map,
      count_visible_block,
      count_rendering_block,
      visible_blocks,
      zrange_x,
      zrange_y,
      rendering_blocks,
      T,
      cam_params);

  if (count_rendering_block != 0)
  {

    ::raycast(
        device_map,
        // device_map.state,
        vertex,
        vertex,
        zrange_x,
        zrange_y,
        T,
        cam_params);
  }
}

void DenseMapping::reset()
{
  device_map.reset();
}

// size_t DenseMapping::fetch_mesh_vertex_only(void *vertex)
// {
//   uint count_triangle = 0;

//   ::create_mesh_vertex_only(
//       device_map
//       device_map.state,
//       count_visible_block,
//       visible_blocks,
//       count_triangle,
//       vertex);

//   return (size_t)count_triangle;
// }

size_t DenseMapping::fetch_mesh_with_normal(void *vertex, void *normal)
{
  uint count_triangle = 0;

  ::create_mesh_with_normal(
      device_map,
      // device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex,
      normal);

  return (size_t)count_triangle;
}

// size_t DenseMapping::fetch_mesh_with_colour(void *vertex, void *colour)
// {
//   uint count_triangle = 0;

//   ::create_mesh_with_colour(
//       device_map
//       device_map.state,
//       count_visible_block,
//       visible_blocks,
//       count_triangle,
//       vertex,
//       colour);

//   return (size_t)count_triangle;
// }
