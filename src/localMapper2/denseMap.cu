#include "localMapper2/denseMap.h"
#include <fstream>

// __device__ MapState param;
// inline void uploadMapState(MapState state)
// {
//     (cudaMemcpyToSymbol(param, &state, sizeof(MapState)));
// }

// MapStruct::MapStruct()
// {
//     state.num_total_buckets_ = 200000;
//     state.num_total_hash_entries_ = 250000;
//     state.num_total_voxel_blocks_ = 200000;
//     state.zmax_raycast = 2.5f;
//     state.zmin_raycast = 0.3f;
//     state.zmax_update = 2.5f;
//     state.zmin_update = 0.3f;
//     state.voxel_size = 0.004f;
//     state.num_max_rendering_blocks_ = 1000000;
//     state.num_max_mesh_triangles_ = 20000000;

//     size.num_blocks = state.num_total_voxel_blocks_;
//     size.num_hash_entries = state.num_total_hash_entries_;
//     size.num_buckets = state.num_total_buckets_;

//     uploadMapState(state);
// }

// MapStruct::MapStruct(MapState state) : state(state)
// {
//     uploadMapState(state);
//     size.num_blocks = state.num_total_voxel_blocks_;
//     size.num_hash_entries = state.num_total_hash_entries_;
//     size.num_buckets = state.num_total_buckets_;
// }

__global__ void resetHashKernel(HashEntry *hash_table, int numEntry)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numEntry)
        return;

    hash_table[index].ptr_ = -1;
    hash_table[index].offset_ = -1;
}

__global__ void resetHeapKernel(int *heap, int *heap_counter, int numBlock)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numBlock)
        return;

    heap[index] = numBlock - index - 1;

    if (index == 0)
    {
        heap_counter[0] = numBlock - 1;
    }
}

void MapStruct::reset()
{

    // dim3 block(1024);
    // dim3 grid(div_up(state.num_total_hash_entries_, block.x));

    // resetHashKernel<<<grid, block>>>(map.hash_table_, state.num_total_hash_entries_);

    // grid = dim3(div_up(state.num_total_voxel_blocks_, block.x));
    // resetHeapKernel<<<grid, block>>>(map.heap_mem_, map.heap_mem_counter_, state.num_total_voxel_blocks_);

    // cudaMemset(map.excess_counter_, 0, sizeof(int));
    // cudaMemset(map.bucket_mutex_, 0, sizeof(int) * state.num_total_buckets_);
    // cudaMemset(map.voxels_, 0, sizeof(Voxel) * state.num_total_voxels());
    dim3 block(1024);
    dim3 grid(div_up(hashTableSize, block.x));
    resetHashKernel<<<grid, block>>>(hash_table_, hashTableSize);

    grid = dim3(div_up(voxelBlockSize, block.x));
    resetHeapKernel<<<grid, block>>>(heap_mem_, heap_mem_counter_, voxelBlockSize);

    cudaMemset(excess_counter_, 0, sizeof(int));
    cudaMemset(bucket_mutex_, 0, sizeof(int) * bucketSize);
    cudaMemset(voxels_, 0, sizeof(Voxel) * voxelBlockSize);
}

// __host__ __device__ int MapState::num_total_voxels() const
// {
//     return num_total_voxel_blocks_ * BLOCK_SIZE3;
// }

// __host__ __device__ float MapState::block_size_metric() const
// {
//     return BLOCK_SIZE * voxel_size;
// }

// __host__ __device__ int MapState::num_total_mesh_vertices() const
// {
//     return 3 * num_max_mesh_triangles_;
// }

// __host__ __device__ float MapState::inverse_voxel_size() const
// {
//     return 1.0f / voxel_size;
// }

// __host__ __device__ int MapState::num_excess_entries() const
// {
//     return num_total_hash_entries_ - num_total_buckets_;
// }

// __host__ __device__ float MapState::truncation_dist() const
// {
//     return 5.0f * voxel_size;
// }

// __host__ __device__ float MapState::raycast_step_scale() const
// {
//     return truncation_dist() * inverse_voxel_size();
// }

// void MapStruct::create()
// {

//     cudaMalloc((void **)&map.excess_counter_, sizeof(int));
//     cudaMalloc((void **)&map.heap_mem_counter_, sizeof(int));
//     cudaMalloc((void **)&map.bucket_mutex_, sizeof(int) * state.num_total_buckets_);
//     cudaMalloc((void **)&map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_);
//     cudaMalloc((void **)&map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_);
//     cudaMalloc((void **)&map.voxels_, sizeof(Voxel) * state.num_total_voxels());
// }

void MapStruct::create(
    int hashTableSize,
    int bucketSize,
    int voxelBlockSize,
    float voxelSize,
    float truncationDist)
{
    cudaMalloc((void **)&excess_counter_, sizeof(int));
    cudaMalloc((void **)&heap_mem_counter_, sizeof(int));
    cudaMalloc((void **)&visibleBlockNum, sizeof(uint));
    cudaMalloc((void **)&bucket_mutex_, sizeof(int) * bucketSize);
    cudaMalloc((void **)&heap_mem_, sizeof(int) * voxelBlockSize);
    cudaMalloc((void **)&hash_table_, sizeof(HashEntry) * hashTableSize);
    cudaMalloc((void **)&visibleTable, sizeof(HashEntry) * hashTableSize);
    cudaMalloc((void **)&voxels_, sizeof(Voxel) * voxelBlockSize * BLOCK_SIZE3);

    this->hashTableSize = hashTableSize;
    this->bucketSize = bucketSize;
    this->voxelBlockSize = voxelBlockSize;
    this->voxelSize = voxelSize;
    this->truncationDist = truncationDist;
}

// void MapStruct::create(MapState map_state)
// {
//     // this->state = map_state;
//     // uploadMapState(state);
//     // create();
// }

void MapStruct::release()
{
    // cudaFree((void *)map.heap_mem_);
    // cudaFree((void *)map.heap_mem_counter_);
    // cudaFree((void *)map.hash_table_);
    // cudaFree((void *)map.bucket_mutex_);
    // cudaFree((void *)map.excess_counter_);
    // cudaFree((void *)map.voxels_);

    cudaFree((void *)heap_mem_);
    cudaFree((void *)heap_mem_counter_);
    cudaFree((void *)hash_table_);
    cudaFree((void *)bucket_mutex_);
    cudaFree((void *)excess_counter_);
    cudaFree((void *)voxels_);
    cudaFree((void *)visibleBlockNum);
    cudaFree((void *)visibleTable);
}

bool MapStruct::empty()
{
    return bucketSize == 0;
}