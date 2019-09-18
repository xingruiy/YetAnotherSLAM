#include "localMapper2/denseMap.h"
#include <fstream>

__device__ MapState param;
inline void uploadMapState(MapState state)
{
    (cudaMemcpyToSymbol(param, &state, sizeof(MapState)));
}

template <bool Device>
MapStruct<Device>::MapStruct()
{
    state.num_total_buckets_ = 200000;
    state.num_total_hash_entries_ = 250000;
    state.num_total_voxel_blocks_ = 200000;
    state.zmax_raycast = 2.5f;
    state.zmin_raycast = 0.3f;
    state.zmax_update = 2.5f;
    state.zmin_update = 0.3f;
    state.voxel_size = 0.004f;
    state.num_max_rendering_blocks_ = 1000000;
    state.num_max_mesh_triangles_ = 20000000;

    size.num_blocks = state.num_total_voxel_blocks_;
    size.num_hash_entries = state.num_total_hash_entries_;
    size.num_buckets = state.num_total_buckets_;

    uploadMapState(state);
}

template <bool Device>
MapStruct<Device>::MapStruct(MapState state) : state(state)
{
    uploadMapState(state);
    size.num_blocks = state.num_total_voxel_blocks_;
    size.num_hash_entries = state.num_total_hash_entries_;
    size.num_buckets = state.num_total_buckets_;
}

#ifdef __CUDACC__
__global__ void reset_hash_entries_kernel(HashEntry *hash_table, int max_num)
{
    const int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index >= max_num)
        return;

    hash_table[index].ptr_ = -1;
    hash_table[index].offset_ = -1;
}

__global__ void reset_heap_memory_kernel(int *heap, int *heap_counter)
{
    const int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index >= param.num_total_voxel_blocks_)
        return;

    heap[index] = param.num_total_voxel_blocks_ - index - 1;

    if (index == 0)
    {
        heap_counter[0] = param.num_total_voxel_blocks_ - 1;
    }
}
#endif

template <bool Device>
void MapStruct<Device>::reset()
{
    if (Device)
    {
#ifdef __CUDACC__
        dim3 thread(1024);
        dim3 block(div_up(state.num_total_hash_entries_, thread.x));

        reset_hash_entries_kernel<<<block, thread>>>(map.hash_table_, state.num_total_hash_entries_);

        block = dim3(div_up(state.num_total_voxel_blocks_, thread.x));
        reset_heap_memory_kernel<<<block, thread>>>(map.heap_mem_, map.heap_mem_counter_);

        (cudaMemset(map.excess_counter_, 0, sizeof(int)));
        (cudaMemset(map.bucket_mutex_, 0, sizeof(int) * state.num_total_buckets_));
        (cudaMemset(map.voxels_, 0, sizeof(Voxel) * state.num_total_voxels()));
#endif
    }
}

__host__ __device__ int MapState::num_total_voxels() const
{
    return num_total_voxel_blocks_ * BLOCK_SIZE3;
}

__host__ __device__ float MapState::block_size_metric() const
{
    return BLOCK_SIZE * voxel_size;
}

__host__ __device__ int MapState::num_total_mesh_vertices() const
{
    return 3 * num_max_mesh_triangles_;
}

__host__ __device__ float MapState::inverse_voxel_size() const
{
    return 1.0f / voxel_size;
}

__host__ __device__ int MapState::num_excess_entries() const
{
    return num_total_hash_entries_ - num_total_buckets_;
}

__host__ __device__ float MapState::truncation_dist() const
{
    return 5.0f * voxel_size;
}

__host__ __device__ float MapState::raycast_step_scale() const
{
    return truncation_dist() * inverse_voxel_size();
}

__device__ bool lockBucket(int *mutex)
{
    if (atomicExch(mutex, 1) != 1)
        return true;
    else
        return false;
}

__device__ void unlockBucket(int *mutex)
{
    atomicExch(mutex, 0);
}

__host__ __device__ int computeHash(const Vec3i &pos, const int &noBuckets)
{
    int res = (pos(0) * 73856093) ^ (pos(1) * 19349669) ^ (pos(2) * 83492791);
    res %= noBuckets;
    return res < 0 ? res + noBuckets : res;
}

__device__ bool deleteHashEntry(int *mem_counter, int *mem, int no_blocks, HashEntry &entry)
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

// __device__ bool deleteHashEntry(MapStorage &map, HashEntry &current)
// {
//     int old = atomicAdd(map.heap_mem_counter_, 1);
//     if (old < param.num_total_voxel_blocks_ - 1)
//     {
//         map.heap_mem_[old + 1] = current.ptr_ / BLOCK_SIZE3;
//         current.ptr_ = -1;
//         return true;
//     }
//     else
//     {
//         atomicSub(map.heap_mem_counter_, 1);
//         return false;
//     }
// }

// __device__ HashEntry createHashEntry(
//     int *memCounter,
//     int *mem,
//     const Vec3i &pos,
//     const int &offset)
// {
//     const int old_val = atomicSub(memCounter, 1);
//     if (old_val >= 0)
//     {
//         const int &ptr = mem[old_val];
//         if (ptr != -1)
//         {
//             return HashEntry(pos, ptr * BLOCK_SIZE3, offset);
//         }
//     }
//     else
//     {
//         atomicAdd(memCounter, 1);
//     }

//     return HashEntry();
// }

__device__ bool createHashEntry(MapStorage &map, const Vec3i &pos, const int &offset, HashEntry *entry)
{
    int old = atomicSub(map.heap_mem_counter_, 1);
    if (old >= 0)
    {
        int ptr = map.heap_mem_[old];
        if (ptr != -1 && entry != nullptr)
        {
            // *entry = HashEntry(pos, ptr * BLOCK_SIZE3, offset);
            entry->pos_ = pos;
            entry->ptr_ = ptr * BLOCK_SIZE3;
            entry->offset_ = offset;
            return true;
        }
    }
    else
    {
        atomicAdd(map.heap_mem_counter_, 1);
    }

    return false;
}

// __device__ inline void createBlock(
//     int *memCounter,
//     int *mem,
//     int *bucketMutex,
//     const int noBucket,
//     int *entryCounter,
//     const int noExcess,
//     HashEntry *hashTable,
//     const Vec3i &blockPos)
// {
//     auto bucketIdx = computeHash(blockPos, noBucket);
//     int *mutex = &bucketMutex[bucketIdx];
//     HashEntry *current = &hashTable[bucketIdx];
//     HashEntry *emptyEntry = NULL;
//     if (current->pos_ == blockPos && current->ptr_ != -1)
//         return;

//     if (current->ptr_ == -1)
//         emptyEntry = current;

//     while (current->offset_ > 0)
//     {
//         bucketIdx = noBucket + current->offset_ - 1;
//         current = &hashTable[bucketIdx];
//         if (current->pos_ == blockPos && current->ptr_ != -1)
//             return;

//         if (current->ptr_ == -1 && !emptyEntry)
//             emptyEntry = current;
//     }

//     if (emptyEntry != NULL)
//     {
//         if (lockBucket(mutex))
//         {
//             *emptyEntry = createHashEntry(memCounter, mem, blockPos, current->offset_);
//             unlockBucket(mutex);
//         }
//     }
//     else
//     {
//         if (lockBucket(mutex))
//         {
//             int offset = atomicAdd(entryCounter, 1);
//             if (offset <= noExcess)
//             {
//                 emptyEntry = &hashTable[noBucket + offset - 1];
//                 *emptyEntry = createHashEntry(memCounter, mem, blockPos, 0);
//                 current->offset_ = offset;
//             }
//             unlockBucket(mutex);
//         }
//     }
// }

__device__ void createBlock(MapStorage &map, const Vec3i &block_pos, int &bucket_index)
{
    bucket_index = computeHash(block_pos, param.num_total_buckets_);
    int *mutex = &map.bucket_mutex_[bucket_index];
    HashEntry *current = &map.hash_table_[bucket_index];
    HashEntry *empty_entry = nullptr;
    if (current->pos_ == block_pos && current->ptr_ != -1)
        return;

    if (current->ptr_ == -1)
        empty_entry = current;

    while (current->offset_ > 0)
    {
        bucket_index = param.num_total_buckets_ + current->offset_ - 1;
        current = &map.hash_table_[bucket_index];
        if (current->pos_ == block_pos && current->ptr_ != -1)
            return;

        if (current->ptr_ == -1 && !empty_entry)
            empty_entry = current;
    }

    if (empty_entry != nullptr)
    {
        if (lockBucket(mutex))
        {
            createHashEntry(map, block_pos, current->offset_, empty_entry);
            unlockBucket(mutex);
        }
    }
    else
    {
        if (lockBucket(mutex))
        {
            int offset = atomicAdd(map.excess_counter_, 1);
            if (offset <= param.num_excess_entries())
            {
                empty_entry = &map.hash_table_[param.num_total_buckets_ + offset - 1];
                if (createHashEntry(map, block_pos, 0, empty_entry))
                    current->offset_ = offset;
            }
            unlockBucket(mutex);
        }
    }
}

__device__ void deleteBlock(MapStorage &map, HashEntry &current)
{
    memset(&map.voxels_[current.ptr_], 0, sizeof(Voxel) * BLOCK_SIZE3);
    int hash_id = computeHash(current.pos_, param.num_total_buckets_);
    int *mutex = &map.bucket_mutex_[hash_id];
    HashEntry *reference = &map.hash_table_[hash_id];
    HashEntry *link_entry = nullptr;

    if (reference->pos_ == current.pos_ && reference->ptr_ != -1)
    {
        if (lockBucket(mutex))
        {
            deleteHashEntry(map.heap_mem_counter_, map.heap_mem_, param.num_total_voxel_blocks_, current);
            unlockBucket(mutex);
            return;
        }
    }
    else
    {
        while (reference->offset_ > 0)
        {
            hash_id = param.num_total_buckets_ + reference->offset_ - 1;
            link_entry = reference;
            reference = &map.hash_table_[hash_id];
            if (reference->pos_ == current.pos_ && reference->ptr_ != -1)
            {
                if (lockBucket(mutex))
                {
                    link_entry->offset_ = current.offset_;
                    deleteHashEntry(map.heap_mem_counter_, map.heap_mem_, param.num_total_voxel_blocks_, current);
                    unlockBucket(mutex);
                    return;
                }
            }
        }
    }
}

__device__ void findEntry(const MapStorage &map, const Vec3i &block_pos, HashEntry *&out)
{
    uint bucket_idx = computeHash(block_pos, param.num_total_buckets_);
    out = &map.hash_table_[bucket_idx];
    if (out->ptr_ != -1 && out->pos_ == block_pos)
        return;

    while (out->offset_ > 0)
    {
        bucket_idx = param.num_total_buckets_ + out->offset_ - 1;
        out = &map.hash_table_[bucket_idx];
        if (out->ptr_ != -1 && out->pos_ == block_pos)
            return;
    }

    out = nullptr;
}

__device__ void findVoxel(const MapStorage &map, const Vec3i &voxel_pos, Voxel *&out)
{
    HashEntry *current;
    findEntry(map, voxelPosToBlockPos(voxel_pos), current);
    if (current != nullptr)
        out = &map.voxels_[current->ptr_ + voxelPosToLocalIdx(voxel_pos)];
}

template <bool Device>
void MapStruct<Device>::create()
{
    if (Device)
    {
#ifdef __CUDACC__
        (cudaMalloc((void **)&map.excess_counter_, sizeof(int)));
        (cudaMalloc((void **)&map.heap_mem_counter_, sizeof(int)));
        (cudaMalloc((void **)&map.bucket_mutex_, sizeof(int) * state.num_total_buckets_));
        (cudaMalloc((void **)&map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_));
        (cudaMalloc((void **)&map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_));
        (cudaMalloc((void **)&map.voxels_, sizeof(Voxel) * state.num_total_voxels()));
#endif
    }
    else
    {
        map.voxels_ = new Voxel[state.num_total_voxels()];
        map.hash_table_ = new HashEntry[state.num_total_hash_entries_];
        map.heap_mem_ = new int[state.num_total_voxel_blocks_];
        map.bucket_mutex_ = new int[state.num_total_buckets_];
        map.heap_mem_counter_ = new int[1];
        map.excess_counter_ = new int[1];
    }
}

template <bool Device>
void MapStruct<Device>::create(MapState map_state)
{
    this->state = map_state;
    uploadMapState(state);
    create();
}

template <bool Device>
void MapStruct<Device>::release()
{
    if (Device)
    {
#ifdef __CUDACC__
        (cudaFree((void *)map.heap_mem_));
        (cudaFree((void *)map.heap_mem_counter_));
        (cudaFree((void *)map.hash_table_));
        (cudaFree((void *)map.bucket_mutex_));
        (cudaFree((void *)map.excess_counter_));
        (cudaFree((void *)map.voxels_));
#endif
    }
    else
    {
        delete[] map.heap_mem_;
        delete[] map.heap_mem_counter_;
        delete[] map.hash_table_;
        delete[] map.bucket_mutex_;
        delete[] map.excess_counter_;
        delete[] map.voxels_;
    }

    std::cout << (Device ? "device" : "host") << " map released." << std::endl;
}

template <bool Device>
void MapStruct<Device>::copyTo(MapStruct<Device> &other) const
{
    if (Device)
    {
#ifdef __CUDACC__
        if (other.empty())
            other.create();

        (cudaMemcpy(other.map.excess_counter_, map.excess_counter_, sizeof(int), cudaMemcpyDeviceToDevice));
        (cudaMemcpy(other.map.heap_mem_counter_, map.heap_mem_counter_, sizeof(int), cudaMemcpyDeviceToDevice));
        (cudaMemcpy(other.map.bucket_mutex_, map.bucket_mutex_, sizeof(int) * state.num_total_buckets_, cudaMemcpyDeviceToDevice));
        (cudaMemcpy(other.map.heap_mem_, map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_, cudaMemcpyDeviceToDevice));
        (cudaMemcpy(other.map.hash_table_, map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_, cudaMemcpyDeviceToDevice));
        (cudaMemcpy(other.map.voxels_, map.voxels_, sizeof(Voxel) * state.num_total_voxels(), cudaMemcpyDeviceToDevice));
#endif
    }
    else
    {
    }
}

template <bool Device>
void MapStruct<Device>::upload(MapStruct<false> &other)
{
    if (!Device)
    {
        exit(0);
    }
    else
    {
#ifdef __CUDACC__
        if (other.empty())
            return;

        (cudaMemcpy(map.excess_counter_, other.map.excess_counter_, sizeof(int), cudaMemcpyHostToDevice));
        (cudaMemcpy(map.heap_mem_counter_, other.map.heap_mem_counter_, sizeof(int), cudaMemcpyHostToDevice));
        (cudaMemcpy(map.bucket_mutex_, other.map.bucket_mutex_, sizeof(int) * state.num_total_buckets_, cudaMemcpyHostToDevice));
        (cudaMemcpy(map.heap_mem_, other.map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_, cudaMemcpyHostToDevice));
        (cudaMemcpy(map.hash_table_, other.map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_, cudaMemcpyHostToDevice));
        (cudaMemcpy(map.voxels_, other.map.voxels_, sizeof(Voxel) * state.num_total_voxels(), cudaMemcpyHostToDevice));
#endif
    }
}

template <bool Device>
void MapStruct<Device>::download(MapStruct<false> &other) const
{
    if (!Device)
    {
        exit(0);
    }
    else
    {
#ifdef __CUDACC__
        if (other.empty())
            other.create();

        (cudaMemcpy(other.map.excess_counter_, map.excess_counter_, sizeof(int), cudaMemcpyDeviceToHost));
        (cudaMemcpy(other.map.heap_mem_counter_, map.heap_mem_counter_, sizeof(int), cudaMemcpyDeviceToHost));
        (cudaMemcpy(other.map.bucket_mutex_, map.bucket_mutex_, sizeof(int) * state.num_total_buckets_, cudaMemcpyDeviceToHost));
        (cudaMemcpy(other.map.heap_mem_, map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_, cudaMemcpyDeviceToHost));
        (cudaMemcpy(other.map.hash_table_, map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_, cudaMemcpyDeviceToHost));
        (cudaMemcpy(other.map.voxels_, map.voxels_, sizeof(Voxel) * state.num_total_voxels(), cudaMemcpyDeviceToHost));
#endif
    }
}

template <bool Device>
bool MapStruct<Device>::empty()
{
    return map.voxels_ == NULL;
}

template <bool Device>
void MapStruct<Device>::exportModel(std::string file_name) const
{
    if (Device)
    {
        return;
    }
}

template <bool Device>
void MapStruct<Device>::writeToDisk(std::string file_name, const bool binary) const
{
    if (Device)
    {
        return;
    }

    std::ofstream file;
    if (binary)
    {
        file.open(file_name, std::ios::out | std::ios::binary);
    }
    else
    {
        file.open(file_name, std::ios::out);
    }

    if (file.is_open())
    {
        file.write((const char *)map.voxels_, sizeof(Voxel) * state.num_total_voxels());
        file.write((const char *)map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_);
        file.write((const char *)map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_);
        file.write((const char *)map.bucket_mutex_, sizeof(int) * state.num_total_buckets_);
        file.write((const char *)map.heap_mem_counter_, sizeof(int));
        file.write((const char *)map.excess_counter_, sizeof(int));
        file.flush();
        std::cout << "file wrote to disk." << std::endl;

        file.close();
    }

    std::ofstream file_param(file_name + ".txt", std::ios::out);
    if (file_param.is_open())
    {
        file_param.write((const char *)&size, sizeof(MapSize));
        file_param.flush();
    }
}

template <bool Device>
void MapStruct<Device>::readFromDisk(std::string file_name, const bool binary)
{
    if (Device)
    {
        return;
    }

    std::ifstream file;
    if (binary)
    {
        file.open(file_name, std::ios::in | std::ios::binary);
    }
    else
    {
        file.open(file_name, std::ios::in);
    }

    if (file.is_open())
    {
        file.read((char *)map.voxels_, sizeof(Voxel) * state.num_total_voxels());
        file.read((char *)map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_);
        file.read((char *)map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_);
        file.read((char *)map.bucket_mutex_, sizeof(int) * state.num_total_buckets_);
        file.read((char *)map.heap_mem_counter_, sizeof(int));
        file.read((char *)map.excess_counter_, sizeof(int));
        std::cout << "file read from disk." << std::endl;
    }

    std::ifstream file_param(file_name + ".txt", std::ios::in);
    if (file_param.is_open())
    {
        file_param.read((char *)&size, sizeof(MapSize));
        file_param.close();
    }
    else
    {
        std::cout << "Read parameters failed." << std::endl;
    }
}

__host__ __device__ Vec3i worldPtToVoxelPos(Vec3f pt, const float &voxelSize)
{
    pt = pt / voxelSize;
    return floor(pt);
}

__host__ __device__ Vec3f voxelPosToWorldPt(const Vec3i &voxelPos, const float &voxelSize)
{
    return voxelPos.cast<float>() * voxelSize;
}

__host__ __device__ Vec3i voxelPosToBlockPos(Vec3i voxelPos)
{
    if (voxelPos(0) < 0)
        voxelPos(0) -= BLOCK_SIZE_SUB_1;
    if (voxelPos(1) < 0)
        voxelPos(1) -= BLOCK_SIZE_SUB_1;
    if (voxelPos(2) < 0)
        voxelPos(2) -= BLOCK_SIZE_SUB_1;

    return voxelPos / BLOCK_SIZE;
}

__host__ __device__ Vec3i blockPosToVoxelPos(const Vec3i &blockPos)
{
    return blockPos * BLOCK_SIZE;
}

__host__ __device__ Vec3i voxelPosToLocalPos(Vec3i voxelPos)
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

__host__ __device__ int localPosToLocalIdx(const Vec3i &localPos)
{
    return localPos(2) * BLOCK_SIZE * BLOCK_SIZE + localPos(1) * BLOCK_SIZE + localPos(0);
}

__host__ __device__ Vec3i localIdxToLocalPos(const int &localIdx)
{
    uint x = localIdx % BLOCK_SIZE;
    uint y = localIdx % (BLOCK_SIZE * BLOCK_SIZE) / BLOCK_SIZE;
    uint z = localIdx / (BLOCK_SIZE * BLOCK_SIZE);
    return Vec3i(x, y, z);
}

__host__ __device__ int voxelPosToLocalIdx(const Vec3i &voxelPos)
{
    return localPosToLocalIdx(voxelPosToLocalPos(voxelPos));
}

template class MapStruct<true>;
template class MapStruct<false>;
