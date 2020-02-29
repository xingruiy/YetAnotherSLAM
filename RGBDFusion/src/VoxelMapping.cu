#include "MappingUtils.h"
#include "VoxelMapping.h"

VoxelMapping::VoxelMapping(const int w, const int h, const Eigen::Matrix3f &K)
    : mK(K.cast<double>())
{
  deviceMap.create(80000, 40000, 40000, 0.006f, 0.02f);
  deviceMap.reset();
  zRangeX.create(h / 8, w / 8, CV_32FC1);
  zRangeY.create(h / 8, w / 8, CV_32FC1);

  mpMeshEngine = new MeshEngine(20000000);
  deviceMap.setMeshEngine(mpMeshEngine);

  cudaMalloc((void **)&mplRenderingBlock, sizeof(RenderingBlock) * 100000);
}

VoxelMapping::~VoxelMapping()
{
  deviceMap.release();
  SafeCall(cudaFree((void **)&mplRenderingBlock));
}

void VoxelMapping::FuseFrame(cv::cuda::GpuMat depth, const Sophus::SE3d &T)
{
  mNumVisibleBlocks = 0;

  fuseDepth(deviceMap, depth, T, mK, mNumVisibleBlocks);
}

void VoxelMapping::TracingDepth(cv::cuda::GpuMat &vertex, const Sophus::SE3d &T)
{
  if (mNumVisibleBlocks == 0)
    return;

  create_rendering_blocks(deviceMap, mNumVisibleBlocks, mNumRenderingBlocks, zRangeX, zRangeY, mplRenderingBlock, T, mK);

  if (mNumRenderingBlocks != 0)
  {
    raycast(deviceMap, vertex, zRangeX, zRangeY, T, mK);
  }
}

void VoxelMapping::reset()
{
  deviceMap.reset();
}

void VoxelMapping::TryMeshification()
{
  deviceMap.UpdateMesh();
}

MapStruct *VoxelMapping::GetMapStruct()
{
  return &deviceMap;
}
