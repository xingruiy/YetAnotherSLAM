#include "MappingUtils.h"
#include "DenseMapping.h"

DenseMapping::DenseMapping(const int w, const int h, const Eigen::Matrix3f &K)
    : mK(K.cast<double>())
{
  deviceMap.create(80000, 40000, 40000, 0.006f, 0.02f);
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

void DenseMapping::fuseFrame(cv::cuda::GpuMat depth, const Sophus::SE3d &T)
{
  count_visible_block = 0;

  ::fuseDepth(
      deviceMap,
      depth,
      T,
      mK,
      count_visible_block);
}

void DenseMapping::raytrace(cv::cuda::GpuMat &vertex, const Sophus::SE3d &T)
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
      mK);

  if (count_rendering_block != 0)
  {

    ::raycast(
        deviceMap,
        vertex,
        zRangeX,
        zRangeY,
        T,
        mK);
  }
}

void DenseMapping::reset()
{
  deviceMap.reset();
}

size_t DenseMapping::fetchMeshWithNormal(void *vertex, void *normal)
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
