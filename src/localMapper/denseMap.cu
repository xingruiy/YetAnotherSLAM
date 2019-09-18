#include "localMapper/denseMap.h"
#include <fstream>

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
    dim3 block(1024);
    dim3 grid(div_up(hashTableSize, block.x));
    resetHashKernel<<<grid, block>>>(hash_table_, hashTableSize);

    grid = dim3(div_up(voxelBlockSize, block.x));
    resetHeapKernel<<<grid, block>>>(heap_mem_, heap_mem_counter_, voxelBlockSize);

    cudaMemset(excess_counter_, 0, sizeof(int));
    cudaMemset(bucket_mutex_, 0, sizeof(int) * bucketSize);
    cudaMemset(voxels_, 0, sizeof(Voxel) * voxelBlockSize);
}

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

void MapStruct::release()
{
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