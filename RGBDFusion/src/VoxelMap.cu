#include "VoxelMap.h"
#include <fstream>

__global__ void resetHashKernel(HashEntry *mplHashTable, int numEntry)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numEntry)
        return;

    mplHashTable[index].ptr = -1;
    mplHashTable[index].offset = -1;
}

__global__ void resetHeapKernel(int *mplHeap, int *mplHeapPtr, int numBlock)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numBlock)
        return;

    if (index == 0)
        mplHeapPtr[0] = numBlock - 1;

    mplHeap[index] = numBlock - index - 1;
}

void MapStruct::reset()
{
    dim3 block(1024);
    dim3 grid(cv::divUp(hashTableSize, block.x));
    resetHashKernel<<<grid, block>>>(mplHashTable, hashTableSize);

    grid = dim3(cv::divUp(voxelBlockSize, block.x));
    resetHeapKernel<<<grid, block>>>(mplHeap, mplHeapPtr, voxelBlockSize);

    cudaMemset(mpLinkedListHead, 0, sizeof(int));
    cudaMemset(mplBucketMutex, 0, sizeof(int) * bucketSize);
    cudaMemset(mplVoxelBlocks, 0, sizeof(Voxel) * BlockSize3 * voxelBlockSize);
}

void MapStruct::create(
    int hashTableSize,
    int bucketSize,
    int voxelBlockSize,
    float voxelSize,
    float truncationDist)
{
    cudaMalloc((void **)&mpLinkedListHead, sizeof(int));
    cudaMalloc((void **)&mplHeapPtr, sizeof(int));
    cudaMalloc((void **)&visibleBlockNum, sizeof(uint));
    cudaMalloc((void **)&mplBucketMutex, sizeof(int) * bucketSize);
    cudaMalloc((void **)&mplHeap, sizeof(int) * voxelBlockSize);
    cudaMalloc((void **)&mplHashTable, sizeof(HashEntry) * hashTableSize);
    cudaMalloc((void **)&visibleTable, sizeof(HashEntry) * hashTableSize);
    cudaMalloc((void **)&mplVoxelBlocks, sizeof(Voxel) * voxelBlockSize * BlockSize3);

    this->hashTableSize = hashTableSize;
    this->bucketSize = bucketSize;
    this->voxelBlockSize = voxelBlockSize;
    this->voxelSize = voxelSize;
    this->truncationDist = truncationDist;
}

void MapStruct::release()
{
    cudaFree((void *)mplHeap);
    cudaFree((void *)mplHeapPtr);
    cudaFree((void *)mplHashTable);
    cudaFree((void *)mplBucketMutex);
    cudaFree((void *)mpLinkedListHead);
    cudaFree((void *)mplVoxelBlocks);
    cudaFree((void *)visibleBlockNum);
    cudaFree((void *)visibleTable);
}

void MapStruct::getVisibleBlockCount(uint &hostData)
{
    cudaMemcpy(&hostData, visibleBlockNum, sizeof(uint), cudaMemcpyDeviceToHost);
}

void MapStruct::resetVisibleBlockCount()
{
    cudaMemset(visibleBlockNum, 0, sizeof(uint));
}

bool MapStruct::empty()
{
    return bucketSize == 0;
}

void MapStruct::UpdateMesh()
{
    if (!mbHasMesh && mpMeshEngine)
    {
        mpMeshEngine->Meshify(this);
        SafeCall(cudaDeviceSynchronize());
        SafeCall(cudaGetLastError());

        mbHasMesh = true;
    }
}

void MapStruct::setMeshEngine(MeshEngine *pMeshEngine)
{
    mpMeshEngine = pMeshEngine;
}
