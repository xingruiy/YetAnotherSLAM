#pragma once
#include "utils/numType.h"
#include <cuda_runtime_api.h>

#define HashEntryEmpty 0
#define HashEntryBusy -1
#define BlockSize 8
#define BlockSizeSub1 7
#define BlockSize3 512

struct Voxel
{
    short sdf;
    char wt;
    Vec3b rgb;
};

struct HashEntry
{
    int ptr;
    int offset;
    Vec3i pos;
};

struct VoxelMap
{
    int *memStack;
    int *stackPtr;
    int *llPtr;
    int *bucketMutex;
    Voxel *voxels;
    HashEntry *hashTable;

    int numBlocks;
    int numEntries;
    int numBuckets;
    float voxelSize;
    float truncationDist;

    VoxelMap(int numEntries, int numBuckets, int numBlocks);
    void allocate();
    void release();
    void reset();
    void writeToDisk(const char *filePath);
    void readFromDisk(const char *filePath);
};