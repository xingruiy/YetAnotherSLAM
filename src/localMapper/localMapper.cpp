#include "localMapper/mapFunctors.h"
#include "localMapper/localMapper.h"

DenseMapping::DenseMapping(int w, int h, Mat33d &K)
    : intrinsics(K)
{
  deviceMap.create(100000, 80000, 120000, 0.005f, 0.02f);
  deviceMap.reset();
  zRangeX.create(h / 8, w / 8, CV_32FC1);
  zRangeY.create(h / 8, w / 8, CV_32FC1);

  cudaMalloc((void **)&listRenderingBlock, sizeof(RenderingBlock) * 100000);
}

DenseMapping::~DenseMapping()
{
  deviceMap.release();
  cudaFree((void **)&listRenderingBlock);
}

void DenseMapping::fuseFrame(GMat depth, const SE3 &T)
{
  count_visible_block = 0;

  ::fuseDepth(
      deviceMap,
      depth,
      T,
      intrinsics,
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
      zRangeX,
      zRangeY,
      listRenderingBlock,
      T,
      intrinsics);

  if (count_rendering_block != 0)
  {

    ::raycast(
        deviceMap,
        vertex,
        zRangeX,
        zRangeY,
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
      count_visible_block,
      count_triangle,
      vertex,
      normal);

  return (size_t)count_triangle;
}
