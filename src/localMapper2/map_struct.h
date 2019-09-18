#pragma once
#include <iostream>
#include "matrix_type.h"
#include "vector_type.h"

#define BLOCK_SIZE 8
#define BLOCK_SIZE3 512
#define BLOCK_SIZE_SUB_1 7

struct HashEntry
{
    __host__ __device__ inline HashEntry();
    __host__ __device__ inline HashEntry(Vector3i pos, int ptr, int offset);
    __host__ __device__ inline HashEntry(const HashEntry &);
    __host__ __device__ inline HashEntry &operator=(const HashEntry &);
    __host__ __device__ inline bool operator==(const Vector3i &) const;
    __host__ __device__ inline bool operator==(const HashEntry &) const;

    int ptr_;
    int offset_;
    Vector3i pos_;
};

__host__ __device__ inline HashEntry::HashEntry()
    : ptr_(-1), offset_(-1)
{
}

__host__ __device__ inline HashEntry::HashEntry(Vector3i pos, int ptr, int offset)
    : pos_(pos), ptr_(ptr), offset_(offset)
{
}

__host__ __device__ inline HashEntry::HashEntry(const HashEntry &H)
    : pos_(H.pos_), ptr_(H.ptr_), offset_(H.offset_)
{
}

__host__ __device__ inline HashEntry &HashEntry::operator=(const HashEntry &H)
{
    pos_ = H.pos_;
    ptr_ = H.ptr_;
    offset_ = H.offset_;
    return *this;
}

__host__ __device__ inline bool HashEntry::operator==(const Vector3i &pos_) const
{
    return this->pos_ == pos_;
}

__host__ __device__ inline bool HashEntry::operator==(const HashEntry &other) const
{
    return other.pos_ == pos_;
}

struct Voxel
{
    __host__ __device__ inline Voxel();
    __host__ __device__ inline Voxel(float sdf, float weight, Vector3c rgb);
    __host__ __device__ inline float getSDF() const;
    __host__ __device__ inline void setSDF(float val);
    __host__ __device__ inline float getWeight() const;
    __host__ __device__ inline void setWeight(float val);

    short sdf;
    float weight;
    Vector3c rgb;
};

__host__ __device__ inline Voxel::Voxel()
    : sdf(0), weight(0), rgb(0)
{
}

__host__ __device__ inline Voxel::Voxel(float sdf, float weight, Vector3c rgb)
    : weight(weight), rgb(rgb)
{
    setSDF(sdf);
}

__host__ __device__ inline float unpackFloat(short val)
{
    return val / (float)32767;
}

__host__ __device__ inline short packFloat(float val)
{
    return (short)(val * 32767);
}

__host__ __device__ inline float Voxel::getSDF() const
{
    return unpackFloat(sdf);
}

__host__ __device__ inline void Voxel::setSDF(float val)
{
    sdf = packFloat(val);
}

__host__ __device__ inline float Voxel::getWeight() const
{
    return weight;
}

__host__ __device__ inline void Voxel::setWeight(float val)
{
    weight = val;
    if (weight > 255)
        weight = 255;
}

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
    Vector2s upper_left;
    Vector2s lower_right;
    Vector2f zrange;
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

__device__ bool createHashEntry(MapStorage &map, const Vector3i &pos, const int &offset, HashEntry *entry);
__device__ bool deleteHashEntry(int *mem_counter, int *mem, int no_blocks, HashEntry &entry);
// __device__ bool deleteHashEntry(MapStorage &map, HashEntry &current);
__device__ void createBlock(MapStorage &map, const Vector3i &blockPos, int &bucket_index);
__device__ void deleteBlock(MapStorage &map, HashEntry &current);
__device__ void findVoxel(const MapStorage &map, const Vector3i &voxel_pos, Voxel *&out);
__device__ void findEntry(const MapStorage &map, const Vector3i &block_pos, HashEntry *&out);

//! Handy functions to modify the map
__host__ __device__ int computeHash(const Vector3i &blockPos, const int &noBuckets);

//! Coordinate converters
__host__ __device__ Vector3i worldPtToVoxelPos(Vector3f pt, const float &voxelSize);
__host__ __device__ Vector3f voxelPosToWorldPt(const Vector3i &voxelPos, const float &voxelSize);
__host__ __device__ Vector3i voxelPosToBlockPos(Vector3i voxelPos);
__host__ __device__ Vector3i blockPosToVoxelPos(const Vector3i &blockPos);
__host__ __device__ Vector3i voxelPosToLocalPos(Vector3i voxelPos);
__host__ __device__ int localPosToLocalIdx(const Vector3i &localPos);
__host__ __device__ Vector3i localIdxToLocalPos(const int &localIdx);
__host__ __device__ int voxelPosToLocalIdx(const Vector3i &voxelPos);
