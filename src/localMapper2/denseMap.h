#pragma once
#include "utils/numType.h"
#include "utils/cudaUtils.h"

#define BLOCK_SIZE 8
#define BLOCK_SIZE3 512
#define BLOCK_SIZE_SUB_1 7

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

__host__ __device__ __forceinline__ Vec3i floor(const Vec3f &pt)
{
    return Vec3i((int)floor(pt(0)), (int)floor(pt(1)), (int)floor(pt(2)));
}

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

// Map info
class MapState
{
public:
    // The total number of buckets in the map
    // NOTE: buckets are allocated for each main entry
    // It dose not cover the excess entries
    int num_total_buckets_;

    // The total number of voxel blocks in the map
    // also determins the size of the heap memory
    // which is used for storing block addresses
    int num_total_voxel_blocks_;

    // The total number of hash entres in the map
    // This is a combination of main entries and
    // the excess entries
    int num_total_hash_entries_;

    int num_max_mesh_triangles_;
    int num_max_rendering_blocks_;

    float zmin_raycast;
    float zmax_raycast;
    float zmin_update;
    float zmax_update;
    float voxel_size;

    __host__ __device__ int num_total_voxels() const;
    __host__ __device__ int num_excess_entries() const;
    __host__ __device__ int num_total_mesh_vertices() const;
    __host__ __device__ float block_size_metric() const;
    __host__ __device__ float inverse_voxel_size() const;
    __host__ __device__ float truncation_dist() const;
    __host__ __device__ float raycast_step_scale() const;
};

struct MapSize
{
    int num_blocks;
    int num_hash_entries;
    int num_buckets;
};

__device__ extern MapState param;

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

template <bool Device>
struct MapStruct
{
    MapStruct();
    MapStruct(MapState param);
    void create();
    void create(MapState param);
    void release();
    bool empty();
    void copyTo(MapStruct<Device> &) const;
    void upload(MapStruct<false> &);
    void download(MapStruct<false> &) const;
    void writeToDisk(std::string, bool binary = true) const;
    void exportModel(std::string) const;
    void readFromDisk(std::string, bool binary = true);
    void reset();

    MapStorage map;
    MapState state;
    MapSize size;
};

__device__ bool createHashEntry(MapStorage &map, const Vec3i &pos, const int &offset, HashEntry *entry);
__device__ bool deleteHashEntry(int *mem_counter, int *mem, int no_blocks, HashEntry &entry);
// __device__ bool deleteHashEntry(MapStorage &map, HashEntry &current);
__device__ void createBlock(MapStorage &map, const Vec3i &blockPos, int &bucket_index);
__device__ void deleteBlock(MapStorage &map, HashEntry &current);
__device__ void findVoxel(const MapStorage &map, const Vec3i &voxel_pos, Voxel *&out);
__device__ void findEntry(const MapStorage &map, const Vec3i &block_pos, HashEntry *&out);

//! Handy functions to modify the map
__host__ __device__ int computeHash(const Vec3i &blockPos, const int &noBuckets);

//! Coordinate converters
__host__ __device__ Vec3i worldPtToVoxelPos(Vec3f pt, const float &voxelSize);
__host__ __device__ Vec3f voxelPosToWorldPt(const Vec3i &voxelPos, const float &voxelSize);
__host__ __device__ Vec3i voxelPosToBlockPos(Vec3i voxelPos);
__host__ __device__ Vec3i blockPosToVoxelPos(const Vec3i &blockPos);
__host__ __device__ Vec3i voxelPosToLocalPos(Vec3i voxelPos);
__host__ __device__ int localPosToLocalIdx(const Vec3i &localPos);
__host__ __device__ Vec3i localIdxToLocalPos(const int &localIdx);
__host__ __device__ int voxelPosToLocalIdx(const Vec3i &voxelPos);
