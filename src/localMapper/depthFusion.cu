#include <opencv2/cudaarithm.hpp>
#include <thrust/device_vector.h>
#include "utils/numType.h"
#include "utils/prefixSum.h"
#include "utils/cudaUtils.h"
#include "localMapper/mapFunctors.h"

__device__ __forceinline__ bool checkVertexVisible(
    Vec3f pt, const SE3f &Tinv,
    const int &cols, const int &rows,
    const float &fx, const float &fy,
    const float &cx, const float &cy,
    const float &depthMin, const float &depthMax)
{
    pt = Tinv * pt;
    Vec2f pt2d = Vec2f(fx * pt(0) / pt(2) + cx, fy * pt(1) / pt(2) + cy);
    return !(pt2d(0) < 0 || pt2d(1) < 0 ||
             pt2d(0) > cols - 1 || pt2d(1) > rows - 1 ||
             pt(2) < depthMin || pt(2) > depthMax);
}

__device__ __forceinline__ bool checkBlockVisible(
    const Vec3i &block_pos,
    const SE3f &Tinv,
    const float &voxelSize,
    const int &cols, const int &rows,
    const float &fx, const float &fy,
    const float &cx, const float &cy,
    const float &depthMin, const float &depthMax)
{
    float scale = voxelSize * BlockSize;
#pragma unroll
    for (int corner = 0; corner < 8; ++corner)
    {
        Vec3i tmp = block_pos;
        tmp(0) += (corner & 1) ? 1 : 0;
        tmp(1) += (corner & 2) ? 1 : 0;
        tmp(2) += (corner & 4) ? 1 : 0;

        if (checkVertexVisible(
                tmp.cast<float>() * scale,
                Tinv,
                cols, rows,
                fx, fy,
                cx, cy,
                depthMin, depthMax))
            return true;
    }

    return false;
}

__global__ void check_visibility_flag_kernel(
    MapStruct map_struct, uchar *flag, SE3f Tinv,
    int cols, int rows, float fx, float fy, float cx, float cy, float voxelSize,
    float depthMin, float depthMax)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= map_struct.hashTableSize)
        return;

    HashEntry &current = map_struct.hash_table_[idx];
    if (current.ptr != -1)
    {
        switch (flag[idx])
        {
        default:
        {
            if (checkBlockVisible(current.pos, Tinv, voxelSize, cols, rows, fx, fy, cx, cy, depthMin, depthMax))
            {
                flag[idx] = 1;
            }
            else
            {
                current.ptr = -1;
                flag[idx] = 0;
            }

            return;
        }
        case 2:
        {
            flag[idx] = 1;
            return;
        }
        }
    }
}

__global__ void copy_visible_block_kernel(HashEntry *hash_table, HashEntry *visible_block, int hashTableSize, const uchar *flag, const int *pos)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= hashTableSize)
        return;

    if (flag[idx] == 1)
        visible_block[pos[idx]] = hash_table[idx];
}

__device__ inline Vec2f project(
    Vec3f pt, float fx, float fy, float cx, float cy)
{
    return Vec2f(fx * pt(0) / pt(2) + cx, fy * pt(1) / pt(2) + cy);
}

__device__ inline Vec3f unproject(
    int x, int y, float z, float invfx, float invfy, float cx, float cy)
{
    return Vec3f(invfx * (x - cx) * z, invfy * (y - cy) * z, z);
}

__device__ inline Vec3f unprojectWorld(
    int x, int y, float z, float invfx,
    float invfy, float cx, float cy, SE3f pose)
{
    return pose * (unproject(x, y, z, invfx, invfy, cx, cy));
}

__global__ void create_blocks_kernel(MapStruct map_struct, cv::cuda::PtrStepSz<float> depth,
                                     float invfx, float invfy, float cx, float cy,
                                     SE3f pose, uchar *flag, float depthMin, float depthMax)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= depth.cols || y >= depth.rows)
        return;

    float z = depth.ptr(y)[x];
    if (isnan(z) || z < depthMin || z > depthMax)
        return;

    float z_thresh = map_struct.truncationDist * 0.5;
    float z_near = max(depthMin, z - z_thresh);
    float z_far = min(depthMax, z + z_thresh);
    if (z_near >= z_far)
        return;

    Vec3i block_near = voxelPosToBlockPos(worldPtToVoxelPos(unprojectWorld(x, y, z_near, invfx, invfy, cx, cy, pose), map_struct.voxelSize));
    Vec3i block_far = voxelPosToBlockPos(worldPtToVoxelPos(unprojectWorld(x, y, z_far, invfx, invfy, cx, cy, pose), map_struct.voxelSize));

    Vec3i d = block_far - block_near;
    Vec3i increment = Vec3i(d(0) < 0 ? -1 : 1, d(1) < 0 ? -1 : 1, d(2) < 0 ? -1 : 1);
    Vec3i absIncrement = Vec3i(abs(d(0)), abs(d(1)), abs(d(2)));
    Vec3i incrementErr = Vec3i(absIncrement(0) << 1, absIncrement(1) << 1, absIncrement(2) << 1);

    int err1;
    int err2;

    // Bresenham's line algorithm
    // details see : https://en.m.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    if ((absIncrement(0) >= absIncrement(1)) && (absIncrement(0) >= absIncrement(2)))
    {
        err1 = incrementErr(1) - 1;
        err2 = incrementErr(2) - 1;
        createBlock(block_near,
                    map_struct.heap_mem_,
                    map_struct.heap_mem_counter_,
                    map_struct.hash_table_,
                    map_struct.bucket_mutex_,
                    map_struct.excess_counter_,
                    map_struct.hashTableSize,
                    map_struct.bucketSize);
        for (int i = 0; i < absIncrement(0); ++i)
        {
            if (err1 > 0)
            {
                block_near(1) += increment(1);
                err1 -= incrementErr(0);
            }

            if (err2 > 0)
            {
                block_near(2) += increment(2);
                err2 -= incrementErr(0);
            }

            err1 += incrementErr(1);
            err2 += incrementErr(2);
            block_near(0) += increment(0);
            createBlock(block_near,
                        map_struct.heap_mem_,
                        map_struct.heap_mem_counter_,
                        map_struct.hash_table_,
                        map_struct.bucket_mutex_,
                        map_struct.excess_counter_,
                        map_struct.hashTableSize,
                        map_struct.bucketSize);
        }
    }
    else if ((absIncrement(1) >= absIncrement(0)) && (absIncrement(1) >= absIncrement(2)))
    {
        err1 = incrementErr(0) - 1;
        err2 = incrementErr(2) - 1;
        createBlock(block_near,
                    map_struct.heap_mem_,
                    map_struct.heap_mem_counter_,
                    map_struct.hash_table_,
                    map_struct.bucket_mutex_,
                    map_struct.excess_counter_,
                    map_struct.hashTableSize,
                    map_struct.bucketSize);
        for (int i = 0; i < absIncrement(1); ++i)
        {
            if (err1 > 0)
            {
                block_near(0) += increment(0);
                err1 -= incrementErr(1);
            }

            if (err2 > 0)
            {
                block_near(2) += increment(2);
                err2 -= incrementErr(1);
            }

            err1 += incrementErr(0);
            err2 += incrementErr(2);
            block_near(1) += increment(1);
            createBlock(block_near,
                        map_struct.heap_mem_,
                        map_struct.heap_mem_counter_,
                        map_struct.hash_table_,
                        map_struct.bucket_mutex_,
                        map_struct.excess_counter_,
                        map_struct.hashTableSize,
                        map_struct.bucketSize);
        }
    }
    else
    {
        err1 = incrementErr(1) - 1;
        err2 = incrementErr(0) - 1;
        createBlock(block_near,
                    map_struct.heap_mem_,
                    map_struct.heap_mem_counter_,
                    map_struct.hash_table_,
                    map_struct.bucket_mutex_,
                    map_struct.excess_counter_,
                    map_struct.hashTableSize,
                    map_struct.bucketSize);
        for (int i = 0; i < absIncrement(2); ++i)
        {
            if (err1 > 0)
            {
                block_near(1) += increment(1);
                err1 -= incrementErr(2);
            }

            if (err2 > 0)
            {
                block_near(0) += increment(0);
                err2 -= incrementErr(2);
            }

            err1 += incrementErr(1);
            err2 += incrementErr(0);
            block_near(2) += increment(2);
            createBlock(block_near,
                        map_struct.heap_mem_,
                        map_struct.heap_mem_counter_,
                        map_struct.hash_table_,
                        map_struct.bucket_mutex_,
                        map_struct.excess_counter_,
                        map_struct.hashTableSize,
                        map_struct.bucketSize);
        }
    }
}

struct CreateBlockLineTracingFunctor
{
    int *heap;
    int *heapPtr;
    HashEntry *hashTable;
    int *bucketMutex;
    int *excessPtr;
    int hashTableSize;
    int bucketSize;

    float voxelSize;
    float truncDistHalf;
    cv::cuda::PtrStepSz<float> depth;

    float invfx, invfy, cx, cy;
    float depthMin, depthMax;

    SE3f T;

    __device__ __forceinline__ void allocateBlock(const Vec3i &blockPos) const
    {
        createBlock(blockPos, heap, heapPtr, hashTable, bucketMutex, excessPtr, hashTableSize, bucketSize);
    }

    __device__ __forceinline__ void operator()() const
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= depth.cols || y >= depth.rows)
            return;

        float dist = depth.ptr(y)[x];
        if (isnan(dist) || dist < depthMin || dist > depthMax)
            return;

        float distNear = max(depthMin, dist - truncDistHalf);
        float distFar = min(depthMax, dist + truncDistHalf);
        if (distNear >= distFar)
            return;

        Vec3i blockStart = voxelPosToBlockPos(worldPtToVoxelPos(unprojectWorld(x, y, distNear, invfx, invfy, cx, cy, T), voxelSize));
        Vec3i blockEnd = voxelPosToBlockPos(worldPtToVoxelPos(unprojectWorld(x, y, distFar, invfx, invfy, cx, cy, T), voxelSize));

        Vec3i dir = blockEnd - blockStart;
        Vec3i increment = Vec3i(dir(0) < 0 ? -1 : 1, dir(1) < 0 ? -1 : 1, dir(2) < 0 ? -1 : 1);
        Vec3i absIncrement = Vec3i(abs(dir(0)), abs(dir(1)), abs(dir(2)));
        Vec3i incrementErr = Vec3i(absIncrement(0) << 1, absIncrement(1) << 1, absIncrement(2) << 1);

        int err1;
        int err2;

        // Bresenham's line algorithm
        // details see : https://en.m.wikipedia.org/wiki/Bresenham%27s_line_algorithm
        if ((absIncrement(0) >= absIncrement(1)) && (absIncrement(0) >= absIncrement(2)))
        {
            err1 = incrementErr(1) - 1;
            err2 = incrementErr(2) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < absIncrement(0); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(1) += increment(1);
                    err1 -= incrementErr(0);
                }

                if (err2 > 0)
                {
                    blockStart(2) += increment(2);
                    err2 -= incrementErr(0);
                }

                err1 += incrementErr(1);
                err2 += incrementErr(2);
                blockStart(0) += increment(0);
                allocateBlock(blockStart);
            }
        }
        else if ((absIncrement(1) >= absIncrement(0)) && (absIncrement(1) >= absIncrement(2)))
        {
            err1 = incrementErr(0) - 1;
            err2 = incrementErr(2) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < absIncrement(1); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(0) += increment(0);
                    err1 -= incrementErr(1);
                }

                if (err2 > 0)
                {
                    blockStart(2) += increment(2);
                    err2 -= incrementErr(1);
                }

                err1 += incrementErr(0);
                err2 += incrementErr(2);
                blockStart(1) += increment(1);
                allocateBlock(blockStart);
            }
        }
        else
        {
            err1 = incrementErr(1) - 1;
            err2 = incrementErr(0) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < absIncrement(2); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(1) += increment(1);
                    err1 -= incrementErr(2);
                }

                if (err2 > 0)
                {
                    blockStart(0) += increment(0);
                    err2 -= incrementErr(2);
                }

                err1 += incrementErr(1);
                err2 += incrementErr(0);
                blockStart(2) += increment(2);
                allocateBlock(blockStart);
            }
        }
    }
};

struct CheckEntryVisibilityFunctor
{
    HashEntry *hashTable;
    HashEntry *visibleEntry;
    uint *visibleEntryCount;
    SE3f Tinv;
    int cols, rows;
    float fx, fy;
    float cx, cy;
    float depthMin;
    float depthMax;
    float voxelSize;
    int hashTableSize;

    __device__ __forceinline__ void operator()() const
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        __shared__ bool needScan;

        if (threadIdx.x == 0)
            needScan = false;

        __syncthreads();

        uint increment = 0;
        if (idx < hashTableSize)
        {
            HashEntry *current = &hashTable[idx];
            if (current->ptr >= 0)
            {
                bool rval = checkBlockVisible(
                    current->pos,
                    Tinv,
                    voxelSize,
                    cols, rows,
                    fx, fy,
                    cx, cy,
                    depthMin,
                    depthMax);

                if (rval)
                {
                    needScan = true;
                    increment = 1;
                }
            }
        }

        __syncthreads();

        if (needScan)
        {
            auto offset = computeOffset<1024>(increment, visibleEntryCount);
            if (offset >= 0 && offset < hashTableSize && idx < hashTableSize)
                visibleEntry[offset] = hashTable[idx];
        }
    }
};

struct DepthFusionFunctor
{

    Voxel *listBlock;
    HashEntry *visible_blocks;

    SE3f Tinv;
    float fx, fy;
    float cx, cy;
    float depthMin;
    float depthMax;

    float truncationDist;
    int hashTableSize;
    float voxelSize;
    uint count_visible_block;

    cv::cuda::PtrStepSz<float> depth;

    __device__ __forceinline__ void operator()() const
    {
        if (blockIdx.x >= hashTableSize || blockIdx.x >= count_visible_block)
            return;

        HashEntry &current = visible_blocks[blockIdx.x];
        if (current.ptr == -1)
            return;

        Vec3i voxelPos = blockPosToVoxelPos(current.pos);

#pragma unroll
        for (int blockIdxZ = 0; blockIdxZ < 8; ++blockIdxZ)
        {
            Vec3i localPos = Vec3i(threadIdx.x, threadIdx.y, blockIdxZ);
            Vec3f pt = Tinv * voxelPosToWorldPt(voxelPos + localPos, voxelSize);

            int u = __float2int_rd(fx * pt(0) / pt(2) + cx + 0.5);
            int v = __float2int_rd(fy * pt(1) / pt(2) + cy + 0.5);
            if (u < 0 || v < 0 || u > depth.cols - 1 || v > depth.rows - 1)
                continue;

            float dist = depth.ptr(v)[u];
            if (isnan(dist) || dist > depthMax || dist < depthMin)
                continue;

            float sdf = dist - pt(2);
            if (sdf < -truncationDist)
                continue;

            sdf = fmin(1.0f, sdf / truncationDist);
            const int localIdx = localPosToLocalIdx(localPos);
            Voxel &voxel = listBlock[current.ptr + localIdx];

            auto oldSDF = unpackFloat(voxel.sdf);
            auto oldWT = voxel.wt;
            auto weight = 1 / dist;

            if (oldWT == 0)
            {
                voxel.sdf = packFloat(sdf);
                voxel.wt = weight;
                continue;
            }

            oldSDF = (oldSDF * oldWT + sdf * weight) / (oldWT + weight);
            voxel.sdf = packFloat(oldSDF);
            voxel.wt = (oldWT + weight);
        }
    }
};

void fuseDepth(
    MapStruct map_struct,
    const GMat depth,
    const SE3 &T,
    const Mat33d &K,
    uint &visible_block_count)
{
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float invfx = 1.0 / K(0, 0);
    float invfy = 1.0 / K(1, 1);

    const int cols = depth.cols;
    const int rows = depth.rows;

    dim3 thread(8, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    CreateBlockLineTracingFunctor bfunctor;
    bfunctor.heap = map_struct.heap_mem_;
    bfunctor.heapPtr = map_struct.heap_mem_counter_;
    bfunctor.hashTable = map_struct.hash_table_;
    bfunctor.bucketMutex = map_struct.bucket_mutex_;
    bfunctor.excessPtr = map_struct.excess_counter_;
    bfunctor.hashTableSize = map_struct.hashTableSize;
    bfunctor.bucketSize = map_struct.bucketSize;
    bfunctor.voxelSize = map_struct.voxelSize;
    bfunctor.truncDistHalf = map_struct.truncationDist * 0.5;
    bfunctor.depth = depth;
    bfunctor.invfx = invfx;
    bfunctor.invfy = invfy;
    bfunctor.cx = cx;
    bfunctor.cy = cy;
    bfunctor.depthMin = 0.1f;
    bfunctor.depthMax = 3.0f;
    bfunctor.T = T.cast<float>();

    callDeviceFunctor<<<block, thread>>>(bfunctor);

    map_struct.resetVisibleBlockCount();

    CheckEntryVisibilityFunctor cfunctor;
    cfunctor.hashTable = map_struct.hash_table_;
    cfunctor.visibleEntry = map_struct.visibleTable;
    cfunctor.visibleEntryCount = map_struct.visibleBlockNum;
    cfunctor.Tinv = T.inverse().cast<float>();
    cfunctor.cols = cols;
    cfunctor.rows = rows;
    cfunctor.fx = fx;
    cfunctor.fy = fy;
    cfunctor.cx = cx;
    cfunctor.cy = cy;
    cfunctor.depthMin = 0.1f;
    cfunctor.depthMax = 3.0f;
    cfunctor.voxelSize = map_struct.voxelSize;
    cfunctor.hashTableSize = map_struct.hashTableSize;

    thread = dim3(1024);
    block = dim3(div_up(map_struct.hashTableSize, thread.x));

    callDeviceFunctor<<<block, thread>>>(cfunctor);

    map_struct.getVisibleBlockCount(visible_block_count);
    if (visible_block_count == 0)
        return;

    DepthFusionFunctor functor;
    functor.listBlock = map_struct.voxels_;
    functor.visible_blocks = map_struct.visibleTable;
    functor.Tinv = T.inverse().cast<float>();
    functor.fx = fx;
    functor.fy = fy;
    functor.cx = cx;
    functor.cy = cy;
    functor.depthMin = 0.1f;
    functor.depthMax = 3.0f;
    functor.truncationDist = map_struct.truncationDist;
    functor.hashTableSize = map_struct.hashTableSize;
    functor.voxelSize = map_struct.voxelSize;
    functor.count_visible_block = visible_block_count;
    functor.depth = depth;

    thread = dim3(8, 8);
    block = dim3(visible_block_count);

    callDeviceFunctor<<<block, thread>>>(functor);
}
