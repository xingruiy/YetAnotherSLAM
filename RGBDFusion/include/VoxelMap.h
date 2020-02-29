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
    bool empty();
    void reset();
    void create(int hashTableSize, int bucketSize, int voxelBlockSize, float voxelSize, float truncationDist);

    void getVisibleBlockCount(uint &hostData);
    void resetVisibleBlockCount();

public:
    MapStruct(const Eigen::Matrix3f &K);
    void setMeshEngine(MeshEngine *pMeshEngine);

    void Release();

    // TODO: create map based on the desired memory space
    MapStruct(int SizeInMB);
    void Create(int SizeInMB);
    int mFootprintInMB;
    // TODO: combine two maps
    void Fuse(MapStruct *pMapStruct);
    // TODO
    void Fuse(cv::cuda::GpuMat depth, const Sophus::SE3d &Tcm);

    // TODO: Save the map to RAM/HardDisk
    void Hibernate();
    void Reactivate();
    bool mbInHibernation;

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

    Sophus::SE3d mTcw;
    Eigen::Matrix3f mK;
};

#endif