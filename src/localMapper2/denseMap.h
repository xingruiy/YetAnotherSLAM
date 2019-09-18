#pragma once
#include "utils/numType.h"
#include "utils/cudaUtils.h"

#define BLOCK_SIZE 8
#define BLOCK_SIZE3 512
#define BLOCK_SIZE_SUB_1 7

struct HashEntry
{
    int ptr_;
    int offset_;
    Vec3i pos_;
};

struct Voxel
{
    short sdf;
    uchar weight;
};

// // Map info
// class MapState
// {
// public:
//     // The total number of buckets in the map
//     // NOTE: buckets are allocated for each main entry
//     // It dose not cover the excess entries
//     int num_total_buckets_;

//     // The total number of voxel blocks in the map
//     // also determins the size of the heap memory
//     // which is used for storing block addresses
//     int num_total_voxel_blocks_;

//     // The total number of hash entres in the map
//     // This is a combination of main entries and
//     // the excess entries
//     int num_total_hash_entries_;

//     int num_max_mesh_triangles_;
//     int num_max_rendering_blocks_;

//     float zmin_raycast;
//     float zmax_raycast;
//     float zmin_update;
//     float zmax_update;
//     float voxel_size;

//     __host__ __device__ int num_total_voxels() const;
//     __host__ __device__ int num_excess_entries() const;
//     __host__ __device__ int num_total_mesh_vertices() const;
//     __host__ __device__ float block_size_metric() const;
//     __host__ __device__ float inverse_voxel_size() const;
//     __host__ __device__ float truncation_dist() const;
//     __host__ __device__ float raycast_step_scale() const;
// };

struct MapSize
{
    int num_blocks;
    int num_hash_entries;
    int num_buckets;
};

// __device__ extern MapState param;

struct RenderingBlock
{
    Vec2s upper_left;
    Vec2s lower_right;
    Vec2f zrange;
};

struct MapStorage
{
    int *heap_mem_;
    int *excess_counter_;
    int *heap_mem_counter_;
    int *bucket_mutex_;
    Voxel *voxels_;
    HashEntry *hash_table_;
};

struct MapStruct
{
    // MapStruct();
    // MapStruct(MapState param);
    // void create();
    // void create(MapState param);
    void release();
    bool empty();
    void reset();

    MapStorage map;
    // MapState state;
    // MapSize size;

    void create(
        int hashTableSize,
        int bucketSize,
        int voxelBlockSize,
        float voxelSize,
        float truncationDist);

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
        voxelPos(0) -= BLOCK_SIZE_SUB_1;
    if (voxelPos(1) < 0)
        voxelPos(1) -= BLOCK_SIZE_SUB_1;
    if (voxelPos(2) < 0)
        voxelPos(2) -= BLOCK_SIZE_SUB_1;

    return voxelPos / BLOCK_SIZE;
}

__host__ __device__ __forceinline__ Vec3i blockPosToVoxelPos(const Vec3i &blockPos)
{
    return blockPos * BLOCK_SIZE;
}

__host__ __device__ __forceinline__ Vec3i voxelPosToLocalPos(Vec3i voxelPos)
{
    int x = voxelPos(0) % BLOCK_SIZE;
    int y = voxelPos(1) % BLOCK_SIZE;
    int z = voxelPos(2) % BLOCK_SIZE;

    if (x < 0)
        x += BLOCK_SIZE;
    if (y < 0)
        y += BLOCK_SIZE;
    if (z < 0)
        z += BLOCK_SIZE;

    return Vec3i(x, y, z);
}

__host__ __device__ __forceinline__ int localPosToLocalIdx(const Vec3i &localPos)
{
    return localPos(2) * BLOCK_SIZE * BLOCK_SIZE + localPos(1) * BLOCK_SIZE + localPos(0);
}

__host__ __device__ __forceinline__ Vec3i localIdxToLocalPos(const int &localIdx)
{
    uint x = localIdx % BLOCK_SIZE;
    uint y = localIdx % (BLOCK_SIZE * BLOCK_SIZE) / BLOCK_SIZE;
    uint z = localIdx / (BLOCK_SIZE * BLOCK_SIZE);
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
        mem[val_old + 1] = entry.ptr_ / BLOCK_SIZE3;
        entry.ptr_ = -1;
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
    HashEntry *entry)
{
    int old = atomicSub(heapPtr, 1);
    if (old >= 0)
    {
        int ptr = heap[old];
        if (ptr != -1 && entry != nullptr)
        {
            entry->pos_ = pos;
            entry->ptr_ = ptr * BLOCK_SIZE3;
            entry->offset_ = offset;
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
    Vec3i &block_pos,
    int *heap,
    int *heapPtr,
    HashEntry *hashTable,
    int *bucketMutex,
    int *excessPtr,
    int hashTableSize,
    int bucketSize)
{
    auto bucket_index = hash(block_pos, bucketSize);
    int *mutex = &bucketMutex[bucket_index];
    HashEntry *current = &hashTable[bucket_index];
    HashEntry *empty_entry = nullptr;
    if (current->pos_ == block_pos && current->ptr_ != -1)
        return;

    if (current->ptr_ == -1)
        empty_entry = current;

    while (current->offset_ > 0)
    {
        bucket_index = bucketSize + current->offset_ - 1;
        current = &hashTable[bucket_index];
        if (current->pos_ == block_pos && current->ptr_ != -1)
            return;

        if (current->ptr_ == -1 && !empty_entry)
            empty_entry = current;
    }

    if (empty_entry != nullptr)
    {
        if (lockBucket(mutex))
        {
            createHashEntry(heap, heapPtr, block_pos, current->offset_, empty_entry);
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
                if (createHashEntry(heap, heapPtr, block_pos, 0, empty_entry))
                    current->offset_ = offset;
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
    memset(&listBlocks[current.ptr_], 0, sizeof(Voxel) * BLOCK_SIZE3);
    int hash_id = hash(current.pos_, bucketSize);
    int *mutex = &bucketMutex[hash_id];
    HashEntry *reference = &hashTable[hash_id];
    HashEntry *link_entry = nullptr;

    if (reference->pos_ == current.pos_ && reference->ptr_ != -1)
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
        while (reference->offset_ > 0)
        {
            hash_id = bucketSize + reference->offset_ - 1;
            link_entry = reference;
            reference = &hashTable[hash_id];
            if (reference->pos_ == current.pos_ && reference->ptr_ != -1)
            {
                if (lockBucket(mutex))
                {
                    link_entry->offset_ = current.offset_;
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
//     const Vec3i &block_pos,
//     HashEntry *&out)
// {
//     uint bucket_idx = hash(block_pos, param.num_total_buckets_);
//     out = &hashTable[bucket_idx];
//     if (out->ptr_ != -1 && out->pos_ == block_pos)
//         return;

//     while (out->offset_ > 0)
//     {
//         bucket_idx = param.num_total_buckets_ + out->offset_ - 1;
//         out = &hashTable[bucket_idx];
//         if (out->ptr_ != -1 && out->pos_ == block_pos)
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
//         out = &map.voxels_[current->ptr_ + voxelPosToLocalIdx(voxelPos)];
// }

__device__ __forceinline__ void findEntry(
    const Vec3i &block_pos,
    HashEntry *&out,
    HashEntry *hashTable,
    int bucketSize)
{
    uint bucket_idx = hash(block_pos, bucketSize);
    out = &hashTable[bucket_idx];
    if (out->ptr_ != -1 && out->pos_ == block_pos)
        return;

    while (out->offset_ > 0)
    {
        bucket_idx = bucketSize + out->offset_ - 1;
        out = &hashTable[bucket_idx];
        if (out->ptr_ != -1 && out->pos_ == block_pos)
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
        out = &listBlocks[current->ptr_ + voxelPosToLocalIdx(voxelPos)];
}

#endif