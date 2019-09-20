#pragma once
#include "utils/numType.h"
#include "utils/cudaUtils.h"

#define BlockSize 8
#define BlockSize3 512
#define BlockSizeSubOne 7

struct HashEntry
{
    int ptr;
    int offset;
    Vec3i pos;
};

struct Voxel
{
    short sdf;
    uchar wt;
};

struct RenderingBlock
{
    Vec2s upper_left;
    Vec2s lower_right;
    Vec2f zrange;
};

struct MapStruct
{
    void release();
    bool empty();
    void reset();
    void create(
        int hashTableSize,
        int bucketSize,
        int voxelBlockSize,
        float voxelSize,
        float truncationDist);

    void getVisibleBlockCount(uint &hostData);
    void resetVisibleBlockCount();

    int bucketSize;
    int hashTableSize;
    int voxelBlockSize;
    float voxelSize;
    float truncationDist;

    int *heap_mem_;
    int *excess_counter_;
    int *heap_mem_counter_;
    int *bucket_mutex_;
    Voxel *voxels_;
    HashEntry *hash_table_;
    HashEntry *visibleTable;
    uint *visibleBlockNum;
};

/*
    Unpack short into float 
*/
__host__ __device__ __forceinline__ float unpackFloat(short val)
{
    return val / (float)SHRT_MAX;
}

/*
    Pack float into short
*/
__host__ __device__ __forceinline__ short packFloat(float val)
{
    return (short)(val * SHRT_MAX);
}

/*
    hash block position
*/
__host__ __device__ __forceinline__ int hash(const Vec3i &pos, const int &noBuckets)
{
    int res = (pos(0) * 73856093) ^ (pos(1) * 19349669) ^ (pos(2) * 83492791);
    res %= noBuckets;
    return res < 0 ? res + noBuckets : res;
}

__host__ __device__ __forceinline__ Vec3i floor(const Vec3f &pt)
{
    return Vec3i((int)floor(pt(0)), (int)floor(pt(1)), (int)floor(pt(2)));
}

__host__ __device__ __forceinline__ Vec3i worldPtToVoxelPos(Vec3f pt, const float &voxelSize)
{
    pt = pt / voxelSize;
    return floor(pt);
}

__host__ __device__ __forceinline__ Vec3f voxelPosToWorldPt(const Vec3i &voxelPos, const float &voxelSize)
{
    return voxelPos.cast<float>() * voxelSize;
}

__host__ __device__ __forceinline__ Vec3i voxelPosToBlockPos(Vec3i voxelPos)
{
    if (voxelPos(0) < 0)
        voxelPos(0) -= BlockSizeSubOne;
    if (voxelPos(1) < 0)
        voxelPos(1) -= BlockSizeSubOne;
    if (voxelPos(2) < 0)
        voxelPos(2) -= BlockSizeSubOne;

    return voxelPos / BlockSize;
}

__host__ __device__ __forceinline__ Vec3i blockPosToVoxelPos(const Vec3i &blockPos)
{
    return blockPos * BlockSize;
}

__host__ __device__ __forceinline__ Vec3i voxelPosToLocalPos(Vec3i voxelPos)
{
    int x = voxelPos(0) % BlockSize;
    int y = voxelPos(1) % BlockSize;
    int z = voxelPos(2) % BlockSize;

    if (x < 0)
        x += BlockSize;
    if (y < 0)
        y += BlockSize;
    if (z < 0)
        z += BlockSize;

    return Vec3i(x, y, z);
}

__host__ __device__ __forceinline__ int localPosToLocalIdx(const Vec3i &localPos)
{
    return localPos(2) * BlockSize * BlockSize + localPos(1) * BlockSize + localPos(0);
}

__host__ __device__ __forceinline__ Vec3i localIdxToLocalPos(const int &localIdx)
{
    uint x = localIdx % BlockSize;
    uint y = localIdx % (BlockSize * BlockSize) / BlockSize;
    uint z = localIdx / (BlockSize * BlockSize);
    return Vec3i(x, y, z);
}

__host__ __device__ __forceinline__ int voxelPosToLocalIdx(const Vec3i &voxelPos)
{
    return localPosToLocalIdx(voxelPosToLocalPos(voxelPos));
}

#ifdef __CUDACC__

__device__ __forceinline__ Vec2f project(
    const Vec3f &pt,
    const float &fx, const float &fy,
    const float &cx, const float &cy)
{
    return Vec2f(fx * pt(0) / pt(2) + cx, fy * pt(1) / pt(2) + cy);
}

__device__ __forceinline__ Vec3f unproject(
    const int &x, const int &y, const float &z,
    const float &invfx, const float &invfy,
    const float &cx, const float &cy)
{
    return Vec3f(invfx * (x - cx) * z, invfy * (y - cy) * z, z);
}

__device__ __forceinline__ Vec3f unprojectWorld(
    const int &x, const int &y, const float &z, const float &invfx,
    const float &invfy, const float &cx, const float &cy, const SE3f &T)
{
    return T * unproject(x, y, z, invfx, invfy, cx, cy);
}

__device__ __forceinline__ bool checkVertexVisible(
    Vec3f pt, const SE3f &Tinv,
    const int &cols, const int &rows,
    const float &fx, const float &fy,
    const float &cx, const float &cy,
    const float &depthMin, const float &depthMax)
{
    pt = Tinv * pt;
    Vec2f pt2d = project(pt, fx, fy, cx, cy);
    return !(pt2d(0) < 0 || pt2d(1) < 0 ||
             pt2d(0) > cols - 1 || pt2d(1) > rows - 1 ||
             pt(2) < depthMin || pt(2) > depthMax);
}

__device__ __forceinline__ bool checkBlockVisible(
    const Vec3i &blockPos,
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
        Vec3i tmp = blockPos;
        tmp(0) += (corner & 1) ? 1 : 0;
        tmp(1) += (corner & 2) ? 1 : 0;
        tmp(2) += (corner & 4) ? 1 : 0;

        if (checkVertexVisible(
                tmp.cast<float>() * scale,
                Tinv,
                cols, rows,
                fx, fy,
                cx, cy,
                depthMin,
                depthMax))

            return true;
    }

    return false;
}

// compare val with the old value stored in *add
// and write the bigger one to *add
__device__ __forceinline__ void atomicMax(float *add, float val)
{
    int *address_as_i = (int *)add;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// compare val with the old value stored in *add
// and write the smaller one to *add
__device__ __forceinline__ void atomicMin(float *add, float val)
{
    int *address_as_i = (int *)add;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

/*
    lock hash bucket
*/
__device__ __forceinline__ bool lockBucket(int *mutex)
{
    if (atomicExch(mutex, 1) != 1)
        return true;
    else
        return false;
}

/*
    unlock hash bucket
*/
__device__ __forceinline__ void unlockBucket(int *mutex)
{
    atomicExch(mutex, 0);
}

__device__ __forceinline__ bool deleteHashEntry(int *heapPtr, int *heap, int voxelBlockSize, HashEntry *entry)
{
    int val_old = atomicAdd(heapPtr, 1);
    if (val_old < voxelBlockSize)
    {
        heap[val_old + 1] = entry->ptr / BlockSize3;
        entry->ptr = -1;
        return true;
    }
    else
    {
        atomicSub(heapPtr, 1);
        return false;
    }
}

__device__ __forceinline__ bool createHashEntry(
    int *heap,
    int *heapPtr,
    const Vec3i &pos,
    const int &offset,
    HashEntry *emptyEntry)
{
    int old = atomicSub(heapPtr, 1);
    if (old >= 0)
    {
        int ptr = heap[old];
        if (ptr != -1 && emptyEntry != NULL)
        {
            emptyEntry->pos = pos;
            emptyEntry->ptr = ptr * BlockSize3;
            emptyEntry->offset = offset;
            return true;
        }
    }
    else
    {
        atomicAdd(heapPtr, 1);
    }

    return false;
}

__device__ __forceinline__ void createBlock(
    const Vec3i &blockPos,
    int *heap,
    int *heapPtr,
    HashEntry *hashTable,
    int *bucketMutex,
    int *excessPtr,
    int hashTableSize,
    int bucketSize)
{
    auto volatileIdx = hash(blockPos, bucketSize);
    int *mutex = &bucketMutex[volatileIdx];
    HashEntry *current = &hashTable[volatileIdx];
    HashEntry *emptyEntry = nullptr;
    if (current->pos == blockPos && current->ptr != -1)
        return;

    if (current->ptr == -1)
        emptyEntry = current;

    while (current->offset >= 0)
    {
        volatileIdx = bucketSize + current->offset - 1;
        current = &hashTable[volatileIdx];
        if (current->pos == blockPos && current->ptr != -1)
            return;

        if (current->ptr == -1 && !emptyEntry)
            emptyEntry = current;
    }

    if (emptyEntry != nullptr)
    {
        if (lockBucket(mutex))
        {
            createHashEntry(heap, heapPtr, blockPos, current->offset, emptyEntry);
            unlockBucket(mutex);
        }
    }
    else
    {
        if (lockBucket(mutex))
        {
            int offset = atomicAdd(excessPtr, 1);
            if ((offset + bucketSize) < hashTableSize)
            {
                emptyEntry = &hashTable[bucketSize + offset - 1];
                if (createHashEntry(heap, heapPtr, blockPos, -1, emptyEntry))
                    current->offset = offset;
            }
            else
                atomicSub(excessPtr, 1);

            unlockBucket(mutex);
        }
    }
}

__device__ __forceinline__ bool findEntry(
    const Vec3i &blockPos,
    HashEntry *&out,
    HashEntry *hashTable,
    int bucketSize)
{
    uint volatileIdx = hash(blockPos, bucketSize);
    out = &hashTable[volatileIdx];
    if (out->ptr != -1 && out->pos == blockPos)
        return true;

    while (out->offset >= 0)
    {
        volatileIdx = bucketSize + out->offset - 1;
        out = &hashTable[volatileIdx];
        if (out->ptr != -1 && out->pos == blockPos)
            return true;
    }

    out = NULL;
    return false;
}

__device__ __forceinline__ void findVoxel(
    const Vec3i &voxelPos,
    Voxel *&out,
    HashEntry *hashTable,
    Voxel *listBlocks,
    int bucketSize)
{
    HashEntry *current;
    if (findEntry(voxelPosToBlockPos(voxelPos), current, hashTable, bucketSize))
        out = &listBlocks[current->ptr + voxelPosToLocalIdx(voxelPos)];
}

#endif