#pragma once
#include "utils/numType.h"
#include "utils/cudaUtils.h"
#include "utils/prefixSum.h"
#include "localMapper/localMap.h"

#define RENDERING_BLOCK_SUBSAMPLE 16

__host__ __device__ __forceinline__ int hash(const Vec3i &blockPos, const int &numBuckets)
{
    int res = (blockPos(0) * 73856093) ^ (blockPos(1) * 19349669) ^ (blockPos(2) * 83492791);
    res %= numBuckets;
    return res < 0 ? res + numBuckets : res;
}

__device__ __forceinline__ bool lockBucket(int *mutex)
{
    if (atomicExch(mutex, HashEntryBusy) != HashEntryBusy)
        return true;
    else
        return false;
}

__device__ __forceinline__ void unlockBucket(int *mutex)
{
    atomicExch(mutex, HashEntryEmpty);
}

__device__ __forceinline__ bool createHashEntry(
    int *memStack,
    int *stackPtr,
    int numBlocks,
    const Vec3i &pos,
    const int &offset,
    HashEntry *entry)
{
    int rval = atomicAdd(stackPtr, 1);
    if (rval < numBlocks)
    {
        int ptr = memStack[rval];
        if (ptr >= 0 && entry != NULL)
        {
            entry->pos = pos;
            entry->ptr = ptr * BlockSize3;
            entry->offset = offset;
            return true;
        }
    }
    else
    {
        atomicSub(stackPtr, 1);
    }

    return false;
}

__device__ __forceinline__ bool deleteHashEntry(int *memStack, int *stackPtr, HashEntry &entry)
{
    int rval = atomicSub(stackPtr, 1);
    if (rval > 0)
    {
        memStack[rval - 1] = entry.ptr / BlockSize3;
        entry.ptr = HashEntryEmpty;
        return true;
    }
    else
    {
        atomicAdd(stackPtr, 1);
        return false;
    }
}

__device__ __forceinline__ void createBlock(
    HashEntry *hashTable,
    int *memStack,
    int *stackPtr,
    int *llPtr,
    int *bucketMutex,
    const int &numBlocks,
    const int &numBuckets,
    const int &numExcessEntries,
    const Vec3i &blockPos)
{
    auto volatileIdx = hash(blockPos, numBuckets);
    HashEntry *volatileEntry = &hashTable[volatileIdx];
    HashEntry *emptyEntry = NULL;

    if (volatileEntry->pos == blockPos && volatileEntry->ptr != HashEntryEmpty)
        return;
    else if (volatileEntry->ptr == HashEntryEmpty)
        emptyEntry = volatileEntry;

    while (emptyEntry == NULL && volatileEntry->offset > 0)
    {
        volatileIdx = volatileEntry->offset + numBuckets - 1;
        volatileEntry = &hashTable[volatileIdx];

        if (volatileEntry->pos == blockPos && volatileEntry->ptr != HashEntryEmpty)
            return;
        else if (volatileEntry->ptr == HashEntryEmpty)
            emptyEntry = volatileEntry;
    }

    if (emptyEntry != NULL)
    {
        auto mutex = &bucketMutex[volatileIdx];
        if (lockBucket(mutex))
        {
            createHashEntry(memStack, stackPtr, numBlocks, blockPos, 0, emptyEntry);
            unlockBucket(mutex);
        }
    }
    else
    {
        auto mutex = &bucketMutex[volatileIdx];
        if (lockBucket(mutex))
        {
            int offset = atomicAdd(llPtr, 1);
            if (offset <= numExcessEntries)
            {
                emptyEntry = &hashTable[numBuckets + offset - 1];
                createHashEntry(memStack, stackPtr, numBlocks, blockPos, 0, emptyEntry);
                emptyEntry->offset = offset;
            }
            unlockBucket(mutex);
        }
    }
}

__device__ __forceinline__ Vec2f project(
    const Vec3f &pt,
    const float &fx, const float &fy,
    const float &cx, const float &cy)
{
    return Vec2f(fx * pt(0) / pt(2) + cx, fy * pt(1) / pt(2) + cy);
}

__device__ __forceinline__ Vec2i projectRound(
    const Vec3f &pt,
    const float &fx, const float &fy,
    const float &cx, const float &cy)
{
    return Vec2i(
        __float2int_rd(fx * pt(0) / pt(2) + cx + 0.5f),
        __float2int_rd(fy * pt(1) / pt(2) + cy + 0.5f));
}

__device__ __forceinline__ Vec3f unproject(
    const int &x, const int &y, const float &z,
    const float &invfx, const float &invfy,
    const float &cx, const float &cy)
{
    return Vec3f(invfx * (x - cx) * z, invfy * (y - cy) * z, z);
}

__device__ __forceinline__ Vec3f unprojectWorld(
    const int &x, const int &y, const float &z,
    const float &invfx, const float &invfy,
    const float &cx, const float &cy,
    const Sophus::SE3f &T)
{
    return T * unproject(x, y, z, invfx, invfy, cx, cy);
}

__device__ __forceinline__ bool isPointVisible(
    const Vec3f &pt, const Sophus::SE3f &TInv,
    float depthMin, float depthMax,
    int cols, int rows, float fx,
    float fy, float cx, float cy)
{
    Vec3f ptTransformed = TInv * pt;
    auto ptWarped = project(ptTransformed, fx, fy, cx, cy);
    return !(ptWarped(0) < 0 || ptWarped(1) < 0 ||
             ptWarped(0) > cols - 1 || ptWarped(1) > rows - 1 ||
             ptTransformed(2) < depthMin || ptTransformed(2) > depthMax);
}

__device__ __forceinline__ bool isBlockVisible(
    const Vec3i &blockPos,
    const Sophus::SE3f &TInv,
    const float &voxelSize,
    const float &depthMin, const float &depthMax,
    const int &cols, const int &rows,
    const float &fx, const float &fy,
    const float &cx, const float &cy)
{
    float scale = voxelSize * BlockSize;
#pragma unroll
    for (int corner = 0; corner < 8; ++corner)
    {
        Vec3i tmp = blockPos;
        tmp(0) += (corner & 1) ? 1 : 0;
        tmp(1) += (corner & 2) ? 1 : 0;
        tmp(2) += (corner & 4) ? 1 : 0;

        if (isPointVisible(tmp.cast<float>() * scale, TInv, depthMin, depthMax, cols, rows, fx, fy, cx, cy))
            return true;
    }

    return false;
}

__device__ __forceinline__ int localVoxelPosToIdx(const Vec3i &localVoxelPos)
{
    return localVoxelPos(2) * BlockSize * BlockSize + localVoxelPos(1) * BlockSize + localVoxelPos(0);
}

__device__ __forceinline__ Vec3i voxelPosToBlockPos(Vec3i voxelPos)
{
    if (voxelPos(0) < 0)
        voxelPos(0) -= BlockSizeSub1;
    if (voxelPos(1) < 0)
        voxelPos(1) -= BlockSizeSub1;
    if (voxelPos(2) < 0)
        voxelPos(2) -= BlockSizeSub1;

    return voxelPos / BlockSize;
}

struct AllocateBlockFunctor
{
    int cols, rows;
    float invfx, invfy, cx, cy;
    float depthMin, depthMax;
    float truncationDistTH;
    float voxelSizeInv;
    cv::cuda::PtrStep<float> depth;

    HashEntry *hashTable;
    int *memStack;
    int *stackPtr;
    int *llPtr;
    int *bucketMutex;
    int numBlocks;
    int numBuckets;
    int numExcessEntries;

    __device__ __forceinline__ void operator()() const
    {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x >= cols && y >= rows)
            return;

        const auto dist = depth.ptr(y)[x];
        if (dist < FLT_EPSILON || dist < depthMin || dist > depthMax)
            return;

        float zNear = fmax(depthMin, dist - truncationDistTH);
        float zFar = fmin(depthMax, dist + truncationDistTH);
        if (zNear >= zNear)
            return;

        Vec3f ptNear = unproject(x, y, zNear, invfx, invfy, cx, cy) * voxelSizeInv;
        Vec3f ptFar = unproject(x, y, zFar, invfx, invfy, cx, cy) * voxelSizeInv;
        Vec3f dir = ptFar - ptNear;

        float length = dir.norm();
        int nSteps = (int)ceil(2.0 * length);
        dir = dir / (float)(nSteps - 1);

        for (int i = 0; i < nSteps; ++i)
        {
            Vec3i blockPos = voxelPosToBlockPos(ptNear.cast<int>());
            createBlock(hashTable, memStack, stackPtr,
                        llPtr, bucketMutex, numBlocks,
                        numBuckets, numExcessEntries, blockPos);
            ptNear += dir;
        }
    }
};

struct CheckVisibilityFunctor
{
    int numHashEntry;
    Sophus::SE3f TInv;

    HashEntry *hashTable;
    HashEntry *visibleEntry;
    uint *numVisibleEntry;

    int cols, rows;
    float fx, fy, cx, cy;
    float voxelSize, depthMin, depthMax;

    __device__ __forceinline__ void operator()() const
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        __shared__ bool needScan;
        if (threadIdx.x == 0)
            needScan = false;

        __syncthreads();

        uint increment = 0;
        if (idx < numHashEntry)
        {
            HashEntry &currentEntry = hashTable[idx];
            if (currentEntry.ptr != HashEntryEmpty)
                if (isBlockVisible(currentEntry.pos, TInv, voxelSize, depthMin, depthMax, cols, rows, fx, fy, cx, cy))
                {
                    needScan = true;
                    increment = 1;
                }
        }

        __syncthreads();

        if (needScan)
        {
            int offset = computeOffset<1024>(increment, numVisibleEntry);
            if (offset != -1 && offset < numHashEntry && idx < numHashEntry)
                visibleEntry[offset] = hashTable[idx];
        }
    }
};

struct DepthFusionFunctor
{
    Sophus::SE3f TInv;
    float voxelSize;
    int cols, rows;
    float depthMin, depthMax;
    float fx, fy, cx, cy;
    int numHashEntry;
    uint numVisibleEntry;
    float truncationDist;
    cv::cuda::PtrStep<float> depth;

    HashEntry *visibleEntry;
    Voxel *voxelBlock;

    __device__ __forceinline__ void operator()() const
    {
        const auto blockId = blockIdx.x;
        if (blockId >= numHashEntry || blockId >= numVisibleEntry)
            return;
        auto entryPtr = visibleEntry[blockId].ptr;
        Vec3i voxelPos = visibleEntry[blockId].pos * BlockSize3;
        const auto truncationDistInv = 1.0f / truncationDist;

#pragma unroll
        for (int blockIdxZ = 0; blockIdxZ < 8; ++blockIdxZ)
        {
            Vec3i localVoxelPos = Vec3i(threadIdx.x, threadIdx.y, blockIdxZ);
            Vec3f ptTransformed = TInv * ((voxelPos + localVoxelPos).cast<float>() * voxelSize);
            auto ptWarped = projectRound(ptTransformed, fx, fy, cx, cy);
            if (ptWarped(0) < 0 || ptWarped(1) < 0 || ptWarped(0) > cols - 1 || ptWarped(1) > rows - 1)
                continue;

            float dist = depth.ptr(ptWarped(1))[ptWarped(0)];
            if (dist < FLT_EPSILON || dist > depthMax || dist < depthMin)
                continue;

            float newSDF = dist - ptTransformed(2);
            if (newSDF < -truncationDist)
                continue;

            newSDF = fmin(1.0f, newSDF * truncationDistInv);
            auto localIdx = localVoxelPosToIdx(localVoxelPos);
            Voxel &voxel = voxelBlock[entryPtr + localIdx];

            auto oldWT = voxel.wt;

            if (oldWT == 0)
            {
                voxel.sdf = newSDF;
                voxel.wt = 1;
            }
            else
            {
                auto oldSDF = voxel.sdf;
                oldSDF = (oldSDF * oldWT + newSDF) / (oldWT + 1);
                voxel.sdf = oldSDF;
                voxel.wt = oldWT + 1;
            }
        }
    }
};

// struct CreateRenderingBlockFunctor
// {
//     int cols, rows;
//     Sophus::SE3f TInv;
//     float fx, fy, cx, cy;
//     float depthMin, depthMax;
//     float voxelSize;

//     uint *numRenderingBlock;
//     uint numVisibleEntry;

//     HashEntry *visibleEntry;
//     RenderingBlock *renderingBlock;
//     mutable cv::cuda::PtrStepSz<Vec2f> zRangeMap;

//     // compare val with the old value stored in *add
//     // and write the bigger one to *add
//     __device__ __forceinline__ void atomicMax(float *add, float val) const
//     {
//         int *address_as_i = (int *)add;
//         int old = *address_as_i, assumed;
//         do
//         {
//             assumed = old;
//             old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
//         } while (assumed != old);
//     }

//     // compare val with the old value stored in *add
//     // and write the smaller one to *add
//     __device__ __forceinline__ void atomicMin(float *add, float val) const
//     {
//         int *address_as_i = (int *)add;
//         int old = *address_as_i, assumed;
//         do
//         {
//             assumed = old;
//             old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
//         } while (assumed != old);
//     }

//     __device__ __forceinline__ bool createRenderingBlock(const Vec3i &blockPos, RenderingBlock &block) const
//     {
//         block.upperLeft = Vec2i(zRangeMap.cols, zRangeMap.rows);
//         block.lowerRight = Vec2i(-1, -1);
//         block.zrange = Vec2f(depthMax, depthMin);

// #pragma unroll
//         for (int corner = 0; corner < 8; ++corner)
//         {
//             Vec3i tmp = blockPos;
//             tmp(0) += (corner & 1) ? 1 : 0;
//             tmp(1) += (corner & 2) ? 1 : 0;
//             tmp(2) += (corner & 4) ? 1 : 0;

//             Vec3f pt = TInv * tmp.cast<float>() * voxelSize * BlockSize;
//             Vec2f ptWarped = project(pt, fx, fy, cx, cy) / RENDERING_BLOCK_SUBSAMPLE;

//             if (block.upperLeft(0) > std::floor(ptWarped(0)))
//                 block.upperLeft(0) = (int)std::floor(ptWarped(0));
//             if (block.lowerRight(0) < ceil(ptWarped(0)))
//                 block.lowerRight(0) = (int)ceil(ptWarped(0));
//             if (block.upperLeft(1) > std::floor(ptWarped(1)))
//                 block.upperLeft(1) = (int)std::floor(ptWarped(1));
//             if (block.lowerRight(1) < ceil(ptWarped(1)))
//                 block.lowerRight(1) = (int)ceil(ptWarped(1));
//             if (block.zrange(0) > pt(2))
//                 block.zrange(0) = pt(2);
//             if (block.zrange(1) < pt(2))
//                 block.zrange(1) = pt(2);
//         }

//         if (block.upperLeft(0) < 0)
//             block.upperLeft(0) = 0;

//         if (block.upperLeft(1) < 0)
//             block.upperLeft(1) = 0;

//         if (block.lowerRight(0) >= zRangeMap.cols)
//             block.lowerRight(0) = zRangeMap.cols - 1;

//         if (block.lowerRight(1) >= zRangeMap.rows)
//             block.lowerRight(1) = zRangeMap.rows - 1;

//         if (block.upperLeft(0) > block.lowerRight(0))
//             return false;

//         if (block.upperLeft(1) > block.lowerRight(1))
//             return false;

//         if (block.zrange(0) < depthMin)
//             block.zrange(0) = depthMin;

//         if (block.zrange(1) < depthMin)
//             return false;

//         return true;
//     }

//     __device__ __forceinline__ void create_rendering_block_list(int offset, const RenderingBlock &block, int &nx, int &ny) const
//     {
//         for (int y = 0; y < ny; ++y)
//         {
//             for (int x = 0; x < nx; ++x)
//             {
//                 if (offset < param.num_max_rendering_blocks_)
//                 {
//                     RenderingBlock &b(rendering_blocks[offset++]);
//                     b.upperLeft(0) = block.upperLeft(0) + x * RENDERING_BLOCK_SIZE_X;
//                     b.upperLeft(1) = block.upperLeft(1) + y * RENDERING_BLOCK_SIZE_Y;
//                     b.lowerRight(0) = block.upperLeft(0) + (x + 1) * RENDERING_BLOCK_SIZE_X;
//                     b.lowerRight(1) = block.upperLeft(1) + (y + 1) * RENDERING_BLOCK_SIZE_Y;

//                     if (b.lowerRight(0) > block.lowerRight(0))
//                         b.lowerRight(0) = block.lowerRight(0);

//                     if (b.lowerRight(1) > block.lowerRight(1))
//                         b.lowerRight(1) = block.lowerRight(1);

//                     b.zrange = block.zrange;
//                 }
//             }
//         }
//     }

//     FUSION_DEVICE inline void operator()() const
//     {
//         int x = threadIdx.x + blockDim.x * blockIdx.x;

//         bool valid = false;
//         uint requiredNoBlocks = 0;
//         RenderingBlock block;
//         int nx, ny;

//         if (x < visible_block_count && visible_block_pos[x].ptr_ != -1)
//         {
//             valid = create_rendering_block(visible_block_pos[x].pos_, block);
//             float dx = (float)block.lowerRight(0) - block.upperLeft(0) + 1;
//             float dy = (float)block.lowerRight(1) - block.upperLeft(1) + 1;
//             nx = __float2int_ru(dx / RENDERING_BLOCK_SIZE_X);
//             ny = __float2int_ru(dy / RENDERING_BLOCK_SIZE_Y);

//             if (valid)
//             {
//                 requiredNoBlocks = nx * ny;
//                 uint totalNoBlocks = *rendering_block_count + requiredNoBlocks;
//                 if (totalNoBlocks >= param.num_max_rendering_blocks_)
//                 {
//                     requiredNoBlocks = 0;
//                 }
//             }
//         }

//         int offset = exclusive_scan<1024>(requiredNoBlocks, rendering_block_count);
//         if (valid && offset != -1 && (offset + requiredNoBlocks) < param.num_max_rendering_blocks_)
//             create_rendering_block_list(offset, block, nx, ny);
//     }

//     FUSION_DEVICE inline void fill_rendering_blocks() const
//     {
//         int x = threadIdx.x;
//         int y = threadIdx.y;

//         int block = blockIdx.x * 4 + blockIdx.y;
//         if (block >= param.num_max_rendering_blocks_)
//             return;

//         RenderingBlock &b(rendering_blocks[block]);

//         int xpos = b.upperLeft(0) + x;
//         if (xpos > b.lowerRight(0) || xpos >= zRangeMap.cols)
//             return;

//         int ypos = b.upperLeft(1) + y;
//         if (ypos > b.lowerRight(1) || ypos >= zRangeMap.rows)
//             return;

//         atomic_min(&zRangeMap.ptr(ypos)[xpos], b.zrange(0));
//         atomic_max(&zrange_y.ptr(ypos)[xpos], b.zrange(1));

//         return;
//     }
// };