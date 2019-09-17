#include "localMapper/denseMap.h"

MapStruct::MapStruct()
    : numEntry(0), numBucket(0),
      numBlock(0), voxelSize(0),
      truncDist(0)
{
}

void MapStruct::reset()
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