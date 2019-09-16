#include "localMapper/localMapper.h"
#include "localMapper/mapFunctor.h"

LocalMapper::LocalMapper(int w, int h, Mat33d &K)
    : frameWidth(w), frameHeight(h), Intrinsics(K)
{
    localMap = std::make_shared<VoxelMap>(30000, 40000, 50000);
}

void LocalMapper::fuseFrame(std::shared_ptr<Frame> frame)
{
    auto depth = GMat(frame->getDepth());
    auto poseInv = frame->getPose().inverse().cast<float>();
    auto cols = depth.cols;
    auto rows = depth.rows;
    uint numVisibleEntry = 0;

    AllocateBlockFunctor allocateBlock;
    allocateBlock.cols = frameWidth;
    allocateBlock.rows = frameHeight;
    allocateBlock.invfx = 1.0 / Intrinsics(0, 0);
    allocateBlock.invfy = 1.0 / Intrinsics(1, 1);
    allocateBlock.cx = Intrinsics(0, 2);
    allocateBlock.cy = Intrinsics(1, 2);
    allocateBlock.depthMin = 0.5f;
    allocateBlock.depthMax = 3.0f;
    allocateBlock.truncationDistTH = localMap->truncationDist * 0.5f;
    allocateBlock.voxelSizeInv = 1.0f / localMap->voxelSize;
    allocateBlock.depth = depth;
    allocateBlock.hashTable = localMap->hashTable;
    allocateBlock.memStack = localMap->memStack;
    allocateBlock.stackPtr = localMap->stackPtr;
    allocateBlock.llPtr = localMap->llPtr;
    allocateBlock.bucketMutex;
    allocateBlock.numBlocks;
    allocateBlock.numBuckets;
    allocateBlock.numExcessEntries;

    dim3 block(8, 8);
    dim3 grid = getGridConfiguration2D(block, cols, rows);
    callDeviceFunctor<<<grid, block>>>(allocateBlock);

    CheckVisibilityFunctor checkVisible;
    checkVisible.TInv = poseInv;
    checkVisible.hashTable = localMap->hashTable;
    checkVisible.numHashEntry = localMap->numEntries;
    checkVisible.visibleEntry;
    checkVisible.numVisibleEntry;
    checkVisible.cols = frameWidth;
    checkVisible.rows = frameHeight;
    checkVisible.fx = Intrinsics(0, 0);
    checkVisible.fy = Intrinsics(1, 1);
    checkVisible.cx = Intrinsics(0, 2);
    checkVisible.cy = Intrinsics(1, 2);
    checkVisible.voxelSize = localMap->voxelSize;
    checkVisible.depthMin = 0.5f;
    checkVisible.depthMax = 3.0f;

    callDeviceFunctor<<<grid, block>>>(allocateBlock);

    DepthFusionFunctor depthFusion;
    depthFusion.TInv = poseInv;
    depthFusion.cols = frameWidth;
    depthFusion.rows = frameHeight;
    depthFusion.depthMin = 0.5f;
    depthFusion.depthMax = 3.0f;
    depthFusion.fx = Intrinsics(0, 0);
    depthFusion.fy = Intrinsics(1, 1);
    depthFusion.cx = Intrinsics(0, 2);
    depthFusion.cy = Intrinsics(1, 2);
    depthFusion.numVisibleEntry = numVisibleEntry;
    depthFusion.depth = depth;

    depthFusion.voxelSize = localMap->voxelSize;
    depthFusion.numHashEntry = localMap->numEntries;
    depthFusion.truncationDist = localMap->truncationDist;
    depthFusion.visibleEntry = localMap->hashTable;
    depthFusion.voxelBlock = localMap->voxels;

    callDeviceFunctor<<<grid, block>>>(depthFusion);
}

void raytracing(Mat &vmap, Mat &image)
{
}

size_t getMesh(float *vertex, float *normal, size_t bufferSize)
{
    return 0;
}