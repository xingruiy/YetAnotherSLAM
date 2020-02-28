#pragma once

#include "CudaUtils.h"
#include <Eigen/Core>
#include <sophus/se3.hpp>

#define BlockSize 8
#define BlockSize3 512
#define BlockSizeSubOne 7

struct HashEntry
{
    int ptr;
    int offset;
    Eigen::Vector3i pos;
};

struct Voxel
{
    short sdf;
    uchar wt;
};

struct RenderingBlock
{
    Eigen::Matrix<short, 2, 1> upper_left;
    Eigen::Matrix<short, 2, 1> lower_right;
    Eigen::Vector2f zrange;
};

struct MapStruct
{
    void release();
    bool empty();
    void reset();
    void create(
        int hashTableSize,
        int bucketSize,
        int voxelBlockSize,
        float voxelSize,
        float truncationDist);

    void getVisibleBlockCount(uint &hostData);
    void resetVisibleBlockCount();

    int bucketSize;
    int hashTableSize;
    int voxelBlockSize;
    float voxelSize;
    float truncationDist;

    int *mplHeap;
    int *mpLinkedListHead;
    int *mplHeapPtr;
    int *mplBucketMutex;
    Voxel *mplVoxelBlocks;
    HashEntry *mplHashTable;
    HashEntry *visibleTable;
    uint *visibleBlockNum;
};
