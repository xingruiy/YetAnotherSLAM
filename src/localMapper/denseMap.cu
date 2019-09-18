#include "localMapper/denseMap.h"
#include "utils/cudaUtils.h"

MapStruct::MapStruct()
    : numEntry(0), numBucket(0),
      numBlock(0), voxelSize(0),
      truncDist(0)
{
}

void MapStruct::create(int numEntry, int numBucket, int numBlock,
                       float voxelSize, float truncationDist)
{
    cudaMalloc((void **)&excessPtr, sizeof(int));
    cudaMalloc((void **)&heapPtr, sizeof(int));
    cudaMalloc((void **)&numVisibleEntry, sizeof(uint));
    cudaMalloc((void **)&bucketMutex, sizeof(int) * numBucket);
    cudaMalloc((void **)&heap, sizeof(int) * numBlock);
    cudaMalloc((void **)&hashTable, sizeof(HashEntry) * numEntry);
    cudaMalloc((void **)&visibleEntry, sizeof(HashEntry) * numEntry);
    cudaMalloc((void **)&voxelBlocks, sizeof(Voxel) * numBlock * BlockSize3);

    this->numEntry = numEntry;
    this->numBlock = numBlock;
    this->numBucket = numBucket;
    this->voxelSize = voxelSize;
    this->truncDist = truncationDist;

    reset();
}

void MapStruct::release()
{
    cudaFree((void *)heap);
    cudaFree((void *)heapPtr);
    cudaFree((void *)hashTable);
    cudaFree((void *)bucketMutex);
    cudaFree((void *)excessPtr);
    cudaFree((void *)voxelBlocks);
    cudaFree((void *)numVisibleEntry);
    cudaFree((void *)visibleEntry);

    this->numEntry = 0;
    this->numBlock = 0;
    this->numBucket = 0;
    this->voxelSize = 0;
    this->truncDist = 0;
}

bool MapStruct::empty()
{
    return numBlock == 0;
}

void MapStruct::resetNumVisibleEntry()
{
    cudaMemset(numVisibleEntry, 0, sizeof(uint));
}

void MapStruct::getNumVisibleEntry(uint &hostData)
{
    cudaMemcpy(&hostData, numVisibleEntry, sizeof(uint), cudaMemcpyDeviceToHost);
}

__global__ void resetHashEntryKernel(HashEntry *hashTable, int numEntry)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= numEntry)
        return;

    hashTable[idx].offset = -1;
    hashTable[idx].ptr = -1;
}

__global__ void resetHeap(int *heap, int *heapPtr, int numBlock)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= numBlock)
        return;

    if (idx == 0)
        heapPtr[0] = numBlock - 1;

    heap[idx] = numBlock - idx - 1;
}

void MapStruct::reset()
{
    dim3 block(1024);
    dim3 grid = getGridConfiguration1D(block, numEntry);

    resetHashEntryKernel<<<grid, block>>>(hashTable, numEntry);

    grid = getGridConfiguration1D(block, numBlock);
    resetHeap<<<grid, block>>>(heap, heapPtr, numBlock);

    cudaMemset(excessPtr, 0, sizeof(uint));
    cudaMemset(bucketMutex, 0, sizeof(int) * numBucket);
    cudaMemset(voxelBlocks, 0, sizeof(Voxel) * numBlock * BlockSize);
}