#include "mapFunctors.h"
#include "DenseMapping.h"

DenseMapping::DenseMapping(int w, int h, const Eigen::Matrix3d &K)
    : mK(K)
{
  deviceMap.create(80000, 40000, 40000, 0.006f, 0.02f);
  deviceMap.reset();
  zRangeX.create(h / 8, w / 8, CV_32FC1);
  zRangeY.create(h / 8, w / 8, CV_32FC1);
  BufferFloat4WxH.create(h, w, CV_32FC4);

  cudaMalloc((void **)&listRenderingBlock, sizeof(RenderingBlock) * 100000);
}

DenseMapping::~DenseMapping()
{
  deviceMap.release();
  cudaFree((void **)&listRenderingBlock);
}

void DenseMapping::fuseFrame(cv::cuda::GpuMat depth, const Sophus::SE3d &T, unsigned int id)
{
  count_visible_block = 0;

  ::fuseDepth(
      deviceMap,
      depth,
      T,
      mK,
      id,
      count_visible_block);
}

void DenseMapping::raytrace(const Sophus::SE3d &T)
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
        BufferFloat4WxH,
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

cv::cuda::GpuMat DenseMapping::GetSyntheticVertexMap()
{
  return BufferFloat4WxH;
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
