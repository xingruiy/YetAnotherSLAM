#include "MappingUtils.h"
#include "VoxelMapping.h"

VoxelMapping::VoxelMapping(const int w, const int h, const Eigen::Matrix3f &K)
    : mK(K.cast<double>())
{
  mpMapStruct = new MapStruct(K);
  mpMapStruct->create(80000, 40000, 40000, 0.006f, 0.02f);
  mpMapStruct->reset();
  zRangeX.create(h / 8, w / 8, CV_32FC1);
  zRangeY.create(h / 8, w / 8, CV_32FC1);

  mpMeshEngine = new MeshEngine(20000000);
  mpMapStruct->setMeshEngine(mpMeshEngine);

  cudaMalloc((void **)&mplRenderingBlock, sizeof(RenderingBlock) * 100000);
}

VoxelMapping::~VoxelMapping()
{
  mpMapStruct->Release();
  SafeCall(cudaFree((void **)&mplRenderingBlock));
}

void VoxelMapping::FuseFrame(cv::cuda::GpuMat depth, const Sophus::SE3d &T)
{
  mNumVisibleBlocks = 0;
  mpMapStruct->Fuse(depth, T);
  // fuseDepth(*mpMapStruct, depth, T, mK, mNumVisibleBlocks);
}

void VoxelMapping::TracingDepth(cv::cuda::GpuMat &vertex, const Sophus::SE3d &T)
{
  if (mNumVisibleBlocks == 0)
    return;

  create_rendering_blocks(*mpMapStruct, mNumVisibleBlocks, mNumRenderingBlocks, zRangeX, zRangeY, mplRenderingBlock, T, mK);

  if (mNumRenderingBlocks != 0)
  {
    raycast(*mpMapStruct, vertex, zRangeX, zRangeY, T, mK);
  }
}

void VoxelMapping::reset()
{
  mpMapStruct->reset();
}

void VoxelMapping::TryMeshification()
{
  mpMapStruct->UpdateMesh();
}

MapStruct *VoxelMapping::GetMapStruct()
{
  return mpMapStruct;
}
