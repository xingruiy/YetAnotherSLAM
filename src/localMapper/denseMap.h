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
    float wt;
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

__device__ __forceinline__ bool deleteHashEntry(int *mem_counter, int *mem, int no_blocks, HashEntry &entry)
{
    int val_old = atomicAdd(mem_counter, 1);
    if (val_old < no_blocks)
    {
        mem[val_old + 1] = entry.ptr / BlockSize3;
        entry.ptr = -1;
        return true;
    }
    else
    {
        atomicSub(mem_counter, 1);
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
    auto bucket_index = hash(blockPos, bucketSize);
    int *mutex = &bucketMutex[bucket_index];
    HashEntry *current = &hashTable[bucket_index];
    HashEntry *empty_entry = nullptr;
    if (current->pos == blockPos && current->ptr != -1)
        return;

    if (current->ptr == -1)
        empty_entry = current;

    while (current->offset > 0)
    {
        bucket_index = bucketSize + current->offset - 1;
        current = &hashTable[bucket_index];
        if (current->pos == blockPos && current->ptr != -1)
            return;

        if (current->ptr == -1 && !empty_entry)
            empty_entry = current;
    }

    if (empty_entry != nullptr)
    {
        if (lockBucket(mutex))
        {
            createHashEntry(heap, heapPtr, blockPos, current->offset, empty_entry);
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
                empty_entry = &hashTable[bucketSize + offset - 1];
                if (createHashEntry(heap, heapPtr, blockPos, 0, empty_entry))
                    current->offset = offset;
            }
            unlockBucket(mutex);
        }
    }
}

__device__ __forceinline__ void deleteBlock(
    HashEntry &current,
    HashEntry *hashTable,
    Voxel *listBlocks,
    int *bucketMutex,
    int *heap,
    int *heapPtr,
    int bucketSize,
    int voxelBlockSize)
{
    memset(&listBlocks[current.ptr], 0, sizeof(Voxel) * BlockSize3);
    int hash_id = hash(current.pos, bucketSize);
    int *mutex = &bucketMutex[hash_id];
    HashEntry *reference = &hashTable[hash_id];
    HashEntry *link_entry = nullptr;

    if (reference->pos == current.pos && reference->ptr != -1)
    {
        if (lockBucket(mutex))
        {
            deleteHashEntry(heapPtr, heap, voxelBlockSize, current);
            unlockBucket(mutex);
            return;
        }
    }
    else
    {
        while (reference->offset > 0)
        {
            hash_id = bucketSize + reference->offset - 1;
            link_entry = reference;
            reference = &hashTable[hash_id];
            if (reference->pos == current.pos && reference->ptr != -1)
            {
                if (lockBucket(mutex))
                {
                    link_entry->offset = current.offset;
                    deleteHashEntry(heapPtr, heap, voxelBlockSize, current);
                    unlockBucket(mutex);
                    return;
                }
            }
        }
    }
}

// __device__ __forceinline__ void findEntry(
//     const MapStorage &map,
//     const Vec3i &blockPos,
//     HashEntry *&out)
// {
//     uint bucket_idx = hash(blockPos, param.num_total_buckets_);
//     out = &hashTable[bucket_idx];
//     if (out->ptr != -1 && out->pos == blockPos)
//         return;

//     while (out->offset > 0)
//     {
//         bucket_idx = param.num_total_buckets_ + out->offset - 1;
//         out = &hashTable[bucket_idx];
//         if (out->ptr != -1 && out->pos == blockPos)
//             return;
//     }

//     out = nullptr;
// }

// __device__ __forceinline__ void findVoxel(
//     const MapStorage &map,
//     const Vec3i &voxelPos, Voxel *&out)
// {
//     HashEntry *current;
//     findEntry(map, voxelPosToBlockPos(voxelPos), current);
//     if (current != nullptr)
//         out = &map.voxels_[current->ptr + voxelPosToLocalIdx(voxelPos)];
// }

__device__ __forceinline__ void findEntry(
    const Vec3i &blockPos,
    HashEntry *&out,
    HashEntry *hashTable,
    int bucketSize)
{
    uint bucket_idx = hash(blockPos, bucketSize);
    out = &hashTable[bucket_idx];
    if (out->ptr != -1 && out->pos == blockPos)
        return;

    while (out->offset > 0)
    {
        bucket_idx = bucketSize + out->offset - 1;
        out = &hashTable[bucket_idx];
        if (out->ptr != -1 && out->pos == blockPos)
            return;
    }

    out = nullptr;
}

__device__ __forceinline__ void findVoxel(
    const Vec3i &voxelPos,
    Voxel *&out,
    HashEntry *hashTable,
    Voxel *listBlocks,
    int bucketSize)
{
    HashEntry *current;
    findEntry(voxelPosToBlockPos(voxelPos), current, hashTable, bucketSize);
    if (current != nullptr)
        out = &listBlocks[current->ptr + voxelPosToLocalIdx(voxelPos)];
}

#endif