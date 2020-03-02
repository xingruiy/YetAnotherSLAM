#ifndef VOXEL_STRUCT_H
#define VOXEL_STRUCT_H

#include <iostream>
#include <mutex>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
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
    void create(int hashTableSize, int bucketSize, int voxelBlockSize, float voxelSize, float truncationDist);

public:
    MapStruct(const Eigen::Matrix3f &K);
    void setMeshEngine(MeshEngine *pMeshEngine);
    void Reset();
    void Release();
    void Swap(MapStruct *pMapStruct);

    // TODO: create map based on the desired memory space
    MapStruct(int SizeInMB);
    void Create(int SizeInMB);
    int mFootPrintInMB;

    // TODO: combine two maps
    void Fuse(MapStruct *pMapStruct);

    // TODO
    void Fuse(cv::cuda::GpuMat depth, const Sophus::SE3d &Tcm);

    // TODO: Save the map to RAM/HardDisk
    void SaveToFile(std::string &strFileName);
    void ReadFromFile(std::string &strFileName);
    void Hibernate();
    void ReActivate();
    bool mbInHibernation;

public:
    void GenerateMesh();
    void DeleteMesh();

    float *mplPoint;
    float *mplNormal;
    int N;
    bool mbActive;
    bool mbHasMesh;

    // OpenGL buffer for Drawing
    float mColourTaint;
    uint mGlVertexBuffer;
    uint mGlNormalBuffer;
    bool mbVertexBufferCreated;

    // Mesh Engine
    MeshEngine *mpMeshEngine;

protected:
    uint mNumVisibleBlocks;

public:
    long unsigned int mnId;
    static long unsigned int nNextId;

    // Heap memory stores the pointers to the actual voxel space.
    // It is in descending order, so the first element has the value N-1.
    int *mplHeap, *mplHeapHib;

    // Heap index indicates the current queue head
    int *mplHeapPtr, *mplHeapPtrHib;

    // Bucket mutex used to lock hash table entries.
    int *mplBucketMutex, *mplBucketMutexHib;

    // Hash table store the entry which points to voxel space
    HashEntry *mplHashTable, *mplHashTableHib;

    // The actual voxel space
    Voxel *mplVoxelBlocks, *mplVoxelBlocksHib;

    int *mpLinkedListHead, *mpLinkedListHeadHib;

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