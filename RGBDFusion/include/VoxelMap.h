#ifndef VOXEL_STRUCT_H
#define VOXEL_STRUCT_H

#include <iostream>
#include <mutex>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "CudaUtils.h"
#include "MeshEngine.h"

#define BlockSize 8
#define BlockSize3 512
#define BlockSizeSubOne 7

class MeshEngine;

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

class MapStruct
{
public:
    void release();
    bool empty();
    void reset();
    void create(int hashTableSize, int bucketSize, int voxelBlockSize, float voxelSize, float truncationDist);

    void getVisibleBlockCount(uint &hostData);
    void resetVisibleBlockCount();

    int GetPointAndNormal(float *pPoint, float *pNormal, const int &N = -1);
    void GetTracingMap(cv::cuda::GpuMat map, const Sophus::SE3d &Tcw);
    void IntegrateDepth(cv::cuda::GpuMat im, cv::cuda::GpuMat depthmap, const Sophus::SE3d &Tcw);

public:
    MapStruct() = default;
    MapStruct(int MemInBytes);
    void Create(int MemInBytes);
    void setMeshEngine(MeshEngine *pMeshEngine);
    void FuseDepth(cv::cuda::GpuMat depth, const Sophus::SE3d &Tcw);

public:
    void UpdateMesh();

    float *mplPoint;
    float *mplNormal;
    int N;
    bool mbActive;
    bool mbHasMesh;
    MeshEngine *mpMeshEngine;

protected:
    uint mNumVisibleBlocks;

public:
    // Heap memory stores the pointers to the actual voxel space.
    // It is in descending order, so the first element has the value N-1.
    int *mplHeap;

    // Heap index indicates the current queue head
    int *mplHeapPtr;

    // Bucket mutex used to lock hash table entries.
    int *mplBucketMutex;

    // Hash table store the entry which points to voxel space
    HashEntry *mplHashTable;

    // The actual voxel space
    Voxel *mplVoxelBlocks;

    int *mpLinkedListHead;

    HashEntry *visibleTable;
    uint *visibleBlockNum;

    int bucketSize;
    int hashTableSize;
    int voxelBlockSize;
    float voxelSize;
    float truncationDist;
};

#endif