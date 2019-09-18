#include "localMapper/localMapper.h"
#include "localMapper/mapFunctor.h"
#include "utils/cudaUtils.h"

LocalMapper::LocalMapper(int w, int h, Mat33d &K)
    : intrinsics(K), frameWidth(w), frameHeight(h)
{
  deviceMap.create(40000, 30000, 20000, 0.005f, 0.02f);
  zrangeX.create(h / RenderingBlockSubSample, w / RenderingBlockSubSample, CV_32FC1);
  zrangeY.create(h / RenderingBlockSubSample, w / RenderingBlockSubSample, CV_32FC1);
  cudaMalloc((void **)&numRenderingBlock, sizeof(uint));
  cudaMalloc((void **)&listRenderingBlock, sizeof(RenderingBlock) * MaxNumRenderingBlock);
}

LocalMapper::~LocalMapper()
{
  deviceMap.release();
  cudaFree(numRenderingBlock);
  cudaFree(listRenderingBlock);
}

void LocalMapper::fuseFrame(GMat depth, const SE3 &T)
{
  preAllocateBlock(depth, T);
  checkBlockInFrustum(T);

  deviceMap.getNumVisibleEntry(numVisibleBlock);
  if (numVisibleBlock == 0)
  {
    printf("ERROR: no visible block, depth fusion failed.\n");
    return;
  }

  DepthFusionFunctor functor;
  functor.depth = depth;
  functor.Tinv = T.inverse().cast<float>();
  functor.cols = depth.cols;
  functor.rows = depth.rows;
  functor.fx = intrinsics(0, 0);
  functor.fy = intrinsics(1, 1);
  functor.cx = intrinsics(0, 2);
  functor.cy = intrinsics(1, 2);
  functor.depthMin = 0.1f;
  functor.depthMax = 3.0f;

  functor.voxels = deviceMap.voxelBlocks;
  functor.visibleEntry = deviceMap.visibleEntry;
  functor.numVisibleEntry = numVisibleBlock;
  functor.numEntry = deviceMap.numEntry;
  functor.truncDist = deviceMap.truncDist;
  functor.voxelSize = deviceMap.voxelSize;

  dim3 block(8, 8);
  dim3 grid(numVisibleBlock);
  callDeviceFunctor<<<grid, block>>>(functor);
  cudaCheckError();
}

void LocalMapper::raytrace(GMat &vertex, const SE3 &T)
{
  if (numVisibleBlock == 0)
  {
    printf("ERROR: no visible block, raytracing failed.\n");
    return;
  }

  projectVisibleBlock(T);

  uint hostData;
  cudaMemcpy(&hostData, numRenderingBlock, sizeof(uint), cudaMemcpyDeviceToHost);

  if (hostData == 0)
  {
    printf("ERROR: could not create rendering blocks, traytrcing failed.\n");
    return;
  }

  predictDepthMap(hostData);

  Mat img(zrangeX);
  cv::resize(img, img, cv::Size(), 8, 8);
  cv::imwrite("img2.png", img);
  // cv::waitKey(1);

  RaytracingFunctor functor;
  functor.zRangeX = zrangeX;
  functor.zRangeY = zrangeY;
  functor.cols = vertex.cols;
  functor.rows = vertex.rows;
  functor.voxelSize = deviceMap.voxelSize;
  functor.voxelSizeInv = 1.0 / deviceMap.voxelSize;
  functor.invfx = 1.0 / intrinsics(0, 0);
  functor.invfy = 1.0 / intrinsics(1, 1);
  functor.cx = intrinsics(0, 2);
  functor.cy = intrinsics(1, 2);
  functor.raycastStep = deviceMap.truncDist / deviceMap.voxelSize;
  functor.T = T.cast<float>();
  functor.Tinv = T.inverse().cast<float>();
  functor.hashTable = deviceMap.hashTable;
  functor.blocks = deviceMap.voxelBlocks;
  functor.numBucket = deviceMap.numBucket;
  functor.vmap = vertex;

  dim3 block(8, 8);
  dim3 grid = getGridConfiguration2D(block, vertex.cols, vertex.rows);
  callDeviceFunctor<<<grid, block>>>(functor);
  cudaCheckError();
}

void LocalMapper::reset()
{
  deviceMap.reset();
}

void LocalMapper::preAllocateBlock(GMat depth, const SE3 &T)
{
  int cols = depth.cols;
  int rows = depth.rows;

  CreateVoxelBlockFunctor functor;
  functor.T = T.cast<float>();
  functor.invfx = 1.0 / intrinsics(0, 0);
  functor.invfy = 1.0 / intrinsics(1, 1);
  functor.cx = intrinsics(0, 2);
  functor.cy = intrinsics(1, 2);
  functor.depthMin = 0.4f,
  functor.depthMax = 3.0f;
  functor.cols = cols,
  functor.rows = rows;
  functor.depth = depth;

  functor.truncDistHalf = deviceMap.truncDist * 0.5f;
  functor.voxelSizeInv = 1.0 / deviceMap.voxelSize;
  functor.numEntry = deviceMap.numEntry;
  functor.numBucket = deviceMap.numBucket;
  functor.hashTable = deviceMap.hashTable;
  functor.bucketMutex = deviceMap.bucketMutex;
  functor.heap = deviceMap.heap;
  functor.heapPtr = deviceMap.heapPtr;
  functor.excessPtr = deviceMap.excessPtr;

  dim3 block(8, 8);
  dim3 grid = getGridConfiguration2D(block, cols, rows);
  callDeviceFunctor<<<grid, block>>>(functor);
  cudaCheckError();
}

void LocalMapper::checkBlockInFrustum(const SE3 &T)
{
  deviceMap.resetNumVisibleEntry();

  CheckEntryVisibilityFunctor functor;
  functor.Tinv = T.inverse().cast<float>();
  functor.cols = frameWidth;
  functor.rows = frameHeight;
  functor.fx = intrinsics(0, 0);
  functor.fy = intrinsics(1, 1);
  functor.cx = intrinsics(0, 2);
  functor.cy = intrinsics(1, 2);
  functor.depthMin = 0.4f;
  functor.depthMax = 3.0f;
  functor.voxelSize = deviceMap.voxelSize;
  functor.numEntry = deviceMap.numEntry;
  functor.hashTable = deviceMap.hashTable;
  functor.visibleEntry = deviceMap.visibleEntry;
  functor.numVisibleEntry = deviceMap.numVisibleEntry;

  dim3 block(1024);
  dim3 grid = getGridConfiguration1D(block, deviceMap.numEntry);
  callDeviceFunctor<<<grid, block>>>(functor);
  cudaCheckError();
}

void LocalMapper::projectVisibleBlock(const SE3 &T)
{
  zrangeX.setTo(cv::Scalar(100.f));
  zrangeY.setTo(cv::Scalar(0));

  cudaMemset(numRenderingBlock, 0, sizeof(uint));

  ProjectBlockFunctor functor;
  functor.fx = intrinsics(0, 0);
  functor.fy = intrinsics(1, 1);
  functor.cx = intrinsics(0, 2);
  functor.cy = intrinsics(1, 2);
  functor.depthMin = 0.4f;
  functor.depthMax = 3.0f;
  functor.zRangeX = zrangeX;
  functor.zRangeY = zrangeY;
  functor.Tinv = T.inverse().cast<float>();
  functor.numVisibleEntry = numVisibleBlock;
  functor.renderingBlock = listRenderingBlock;
  functor.numRenderingBlock = numRenderingBlock;
  functor.scale = deviceMap.voxelSize * BlockSize;
  functor.visibleEntry = deviceMap.visibleEntry;

  dim3 block(1024);
  dim3 grid = getGridConfiguration1D(block, numVisibleBlock);
  callDeviceFunctor<<<grid, block>>>(functor);
  cudaCheckError();
}

void LocalMapper::predictDepthMap(uint renderingBlockNum)
{
  DepthPredictionFunctor functor;
  functor.numRenderingBlock = renderingBlockNum;
  functor.renderingBlock = listRenderingBlock;
  functor.zRangeX = zrangeX;
  functor.zRangeY = zrangeY;

  dim3 block(RenderingBlockSizeX, RenderingBlockSizeY);
  dim3 grid = dim3((uint)ceil((float)renderingBlockNum / 4.f), 4);
  callDeviceFunctor<<<grid, block>>>(functor);
  cudaCheckError();
}