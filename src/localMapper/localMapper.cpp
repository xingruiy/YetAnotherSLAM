#include <cuda_runtime_api.h>
#include "map_proc.h"
#include "localMapper.h"

DenseMapping::DenseMapping(int w, int h, Mat33d &K)
    : intrinsics(K)
{
  deviceMap.create(100000, 80000, 120000, 0.005f, 0.02f);
  deviceMap.reset();
  zrange_x.create(h / 8, w / 8, CV_32FC1);
  zrange_y.create(h / 8, w / 8, CV_32FC1);

  cudaMalloc((void **)&visible_blocks, sizeof(HashEntry) * 100000);
  cudaMalloc((void **)&rendering_blocks, sizeof(RenderingBlock) * 100000);
}

DenseMapping::~DenseMapping()
{
  deviceMap.release();
  cudaFree((void **)&visible_blocks);
  cudaFree((void **)&rendering_blocks);
}

void DenseMapping::fuseFrame(GMat depth, const SE3 &T)
{
  count_visible_block = 0;

  ::update(
      deviceMap,
      depth,
      T,
      intrinsics,
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
      deviceMap,
      count_visible_block,
      count_rendering_block,
      visible_blocks,
      zrange_x,
      zrange_y,
      rendering_blocks,
      T,
      intrinsics);

  if (count_rendering_block != 0)
  {

    ::raycast(
        deviceMap,
        // deviceMap.state,
        vertex,
        vertex,
        zrange_x,
        zrange_y,
        T,
        intrinsics);
  }
}

void DenseMapping::reset()
{
  deviceMap.reset();
}

size_t DenseMapping::fetch_mesh_with_normal(void *vertex, void *normal)
{
  uint count_triangle = 0;

  ::create_mesh_with_normal(
      deviceMap,
      // deviceMap.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex,
      normal);

  return (size_t)count_triangle;
}
