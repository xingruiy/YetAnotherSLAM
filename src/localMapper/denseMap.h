#pragma once
#include "utils/numType.h"
#include <cuda_runtime_api.h>

#define BlockSize 8
#define BlockSize3 512
#define BlockSizeSubOne 7

struct Voxel
{
    short sdf;
    uchar wt;
    Vec3b rgb;
};

struct HashEntry
{
    int ptr;
    int offset;
    Vec3i pos;
};

struct RenderingBlock
{
    Vec2s upperLeft;
    Vec2s lowerRight;
    Vec2f zrange;
};

struct MapStruct
{
    MapStruct();
    void create(
        int numEntry, int numBucket, int numBlock,
        float voxelSize, float truncationDist);
    void release();
    bool empty();
    void reset();

    void resetNumVisibleEntry();
    void getNumVisibleEntry(uint &hostData);

    int *heap;
    int *excessPtr;
    int *heapPtr;
    int *bucketMutex;
    Voxel *voxelBlocks;
    HashEntry *hashTable;

    uint *numVisibleEntry;
    HashEntry *visibleEntry;

    int numEntry;
    int numBucket;
    int numBlock;
    float voxelSize;
    float truncDist;
};

#ifdef __CUDACC__

__host__ __device__ __forceinline__ Vec3i worldPtToVoxelPos(const Vec3f &pt, const float &voxelSize)
{
    Vec3i pos((int)(pt(0) / voxelSize), (int)(pt(1) / voxelSize), (int)(pt(2) / voxelSize));
    pos(0) = pos(0) > pt(0) ? pos(0) - 1 : pos(0);
    pos(1) = pos(1) > pt(1) ? pos(1) - 1 : pos(1);
    pos(2) = pos(2) > pt(2) ? pos(2) - 1 : pos(2);
    return pos;
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

/*
    Unpack short into float 
*/
__host__ __device__ __forceinline__ float unpackFloat(short val)
{
    return val / (float)32767;
}

/*
    Pack float into short
*/
__host__ __device__ __forceinline__ short packFloat(float val)
{
    return (short)(val * 32767);
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

/*
    Remove unused hash entry
*/
__device__ __forceinline__ bool removeHashEntry(int *heapPtr, int *heap, int numBlock, HashEntry &entry)
{
    int old = atomicAdd(heapPtr, 1);
    if (old < numBlock)
    {
        heap[old + 1] = entry.ptr / BlockSize3;
        entry.ptr = -1;
        return true;
    }
    else
    {
        atomicSub(heapPtr, 1);
        return false;
    }
}

/*
    create new hash entry
*/
__device__ __forceinline__ bool createHashEntry(int *heapPtr, int *heap, const Vec3i &pos, const int &offset, HashEntry *emptyEntry)
{
    if (emptyEntry == NULL)
        return false;

    int old = atomicSub(heapPtr, 1);
    if (old >= 0)
    {
        emptyEntry->pos = pos;
        emptyEntry->ptr = heap[old] * BlockSize3;
        emptyEntry->offset = offset;
        return true;
    }
    else
    {
        atomicAdd(heapPtr, 1);
        return false;
    }

    return false;
}

/*
    find hash entry based on given block pos
*/
__device__ __forceinline__ bool findEntry(HashEntry *hashTable, const Vec3i &blockPos, const int numBucket, HashEntry *&out)
{
    auto volatileIdx = hash(blockPos, numBucket);
    out = &hashTable[volatileIdx];
    if (out->pos == blockPos && out->ptr != -1)
        return true;

    while (out->offset >= 0)
    {
        volatileIdx = numBucket + out->offset - 1;
        out = &hashTable[volatileIdx];
        if (out->pos == blockPos && out->ptr != -1)
            return true;
    }

    out = NULL;
    return false;
}

/*
    find voxel based on given voxel pos
*/
__device__ __forceinline__ void findVoxel(HashEntry *hashTable, Voxel *blocks, const int &numBucket, const Vec3i &voxelPos, Voxel *&out)
{
    HashEntry *current;
    if (findEntry(hashTable, voxelPosToBlockPos(voxelPos), numBucket, current))
        out = &blocks[current->ptr + voxelPosToLocalIdx(voxelPos)];
}

/*
    create new block and insert into hash table
*/
__device__ __forceinline__ void createBlock(
    HashEntry *hashTable, int *bucketMutex,
    int *heap, int *heapPtr,
    int *excessPtr, const int numEntry,
    const int numBucket, const Vec3i &blockPos)
{
    // auto volatileIdx = hash(blockPos, numBucket);
    // auto *mutex = &bucketMutex[volatileIdx];
    // HashEntry *lastLookedEntry = &hashTable[volatileIdx];
    // HashEntry *emptyEntry = NULL;
    // if (lastLookedEntry->pos == blockPos && lastLookedEntry->ptr != -1)
    //     return;

    // if (lastLookedEntry->ptr == -1)
    //     emptyEntry = lastLookedEntry;

    // while (lastLookedEntry->offset >= 0)
    // {
    //     volatileIdx = numBucket + lastLookedEntry->offset - 1;
    //     lastLookedEntry = &hashTable[volatileIdx];
    //     if (lastLookedEntry->pos == blockPos && lastLookedEntry->ptr != -1)
    //         return;

    //     if (lastLookedEntry->ptr == -1)
    //         emptyEntry = lastLookedEntry;
    // }

    // if (lockBucket(mutex))
    // {
    //     if (emptyEntry != NULL)
    //         createHashEntry(heapPtr, heap, blockPos, lastLookedEntry->offset, emptyEntry);
    //     else
    //     {
    //         int newOffset = atomicAdd(excessPtr, 1);
    //         if (newOffset + numBucket <= numEntry)
    //         {
    //             emptyEntry = &hashTable[numBucket + newOffset - 1];
    //             if (createHashEntry(heapPtr, heap, blockPos, -1, emptyEntry))
    //                 lastLookedEntry->offset = newOffset;
    //         }
    //     }

    //     unlockBucket(mutex);
    // }
    auto volatileIdx = hash(blockPos, numBucket);
    auto *mutex = &bucketMutex[volatileIdx];
    HashEntry *lastLookedEntry = &hashTable[volatileIdx];
    if (lastLookedEntry->pos == blockPos && lastLookedEntry->ptr != -1)
        return;

    while (lastLookedEntry->offset >= 0)
    {
        volatileIdx = numBucket + lastLookedEntry->offset - 1;
        lastLookedEntry = &hashTable[volatileIdx];
        if (lastLookedEntry->pos == blockPos && lastLookedEntry->ptr != -1)
            return;
    }

    if (lockBucket(mutex))
    {
        int newOffset = atomicAdd(excessPtr, 1);
        if (newOffset + numBucket <= numEntry)
        {
            auto *emptyEntry = &hashTable[numBucket + newOffset - 1];
            if (createHashEntry(heapPtr, heap, blockPos, -1, emptyEntry))
                lastLookedEntry->offset = newOffset;
        }

        unlockBucket(mutex);
    }
}

#endif