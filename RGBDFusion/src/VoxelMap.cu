#include "VoxelMap.h"
#include "CudaUtils.h"
#include "ImageProc.h"
#include "ParallelScan.h"
#include "VoxelStructUtils.h"

#define HASHENTRY_IN_BYTE 0.00002
#define VOXEL_BLOCK_IN_BYTE 0.000003

long unsigned int MapStruct::nNextId = 0;

Sophus::SE3d MapStruct::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTcw;
}

void MapStruct::SetPose(Sophus::SE3d &Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTcw = Tcw;
}

__global__ void ResetHash_kernel(HashEntry *mplHashTable, int numEntry)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numEntry)
        return;

    mplHashTable[index].ptr = -1;
    mplHashTable[index].offset = -1;
}

__global__ void ResetHeap_kernel(int *mplHeap, int *mplHeapPtr, int numBlock)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numBlock)
        return;

    if (index == 0)
        mplHeapPtr[0] = numBlock - 1;

    mplHeap[index] = numBlock - index - 1;
}

void MapStruct::Reset()
{
    dim3 block(1024);
    dim3 grid(cv::divUp(hashTableSize, block.x));
    ResetHash_kernel<<<grid, block>>>(mplHashTable, hashTableSize);

    grid = dim3(cv::divUp(voxelBlockSize, block.x));
    ResetHeap_kernel<<<grid, block>>>(mplHeap, mplHeapPtr, voxelBlockSize);

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
    SafeCall(cudaMalloc((void **)&mpLinkedListHead, sizeof(int)));
    SafeCall(cudaMalloc((void **)&mplHeapPtr, sizeof(int)));
    SafeCall(cudaMalloc((void **)&visibleBlockNum, sizeof(uint)));
    SafeCall(cudaMalloc((void **)&mplBucketMutex, sizeof(int) * bucketSize));
    SafeCall(cudaMalloc((void **)&mplHeap, sizeof(int) * voxelBlockSize));
    SafeCall(cudaMalloc((void **)&mplHashTable, sizeof(HashEntry) * hashTableSize));
    SafeCall(cudaMalloc((void **)&visibleTable, sizeof(HashEntry) * hashTableSize));
    SafeCall(cudaMalloc((void **)&mplVoxelBlocks, sizeof(Voxel) * voxelBlockSize * BlockSize3));

    this->hashTableSize = hashTableSize;
    this->bucketSize = bucketSize;
    this->voxelBlockSize = voxelBlockSize;
    this->voxelSize = voxelSize;
    this->truncationDist = truncationDist;
}

MapStruct::MapStruct(const Eigen::Matrix3f &K)
    : mFootPrintInMB(0), mbInHibernation(false), mbActive(true),
      mbHasMesh(false), mpMeshEngine(nullptr), mplHeap(nullptr),
      mplHeapPtr(nullptr), mplBucketMutex(nullptr), mplHashTable(nullptr),
      mplVoxelBlocks(nullptr), mpLinkedListHead(nullptr), mK(K),
      mbVertexBufferCreated(false)
{
    // Get a random colour taint for visualization
    mColourTaint = 255 * rand() / (double)RAND_MAX;
    mnId = nNextId++;
}

MapStruct::MapStruct(int SizeInMB)
{
    // int nHashEntry = 0;  // 160kb
    // int nVoxelBlock = 0; // 3072kb
    // int nBucket = 0;
    // float voxelSize = 0.005;
    // float TruncationDist = 0.02;
}

void MapStruct::Release()
{
    if (!Empty())
    {
        SafeCall(cudaFree((void *)mplHeap));
        SafeCall(cudaFree((void *)mplHeapPtr));
        SafeCall(cudaFree((void *)mplHashTable));
        SafeCall(cudaFree((void *)mplBucketMutex));
        SafeCall(cudaFree((void *)mpLinkedListHead));
        SafeCall(cudaFree((void *)mplVoxelBlocks));
        SafeCall(cudaFree((void *)visibleBlockNum));
        SafeCall(cudaFree((void *)visibleTable));
    }

    if (mbHasMesh && N > 0)
    {
        N = 0;
        free(mplPoint);
        free(mplNormal);
        mbHasMesh = false;
    }

    mplHeap = nullptr;
    mplHeapPtr = nullptr;
    mplHashTable = nullptr;
    visibleTable = nullptr;
    mplVoxelBlocks = nullptr;
    mplBucketMutex = nullptr;
    visibleBlockNum = nullptr;
    mpLinkedListHead = nullptr;
    mbInHibernation = false;
    bucketSize = 0;
}

bool MapStruct::Empty()
{
    return bucketSize == 0;
}

void MapStruct::GenerateMesh()
{
    if (!mbHasMesh && mpMeshEngine && !mbInHibernation)
    {
        mpMeshEngine->Meshify(this);
        mbHasMesh = true;
    }
}

void MapStruct::DeleteMesh()
{
    if (mbHasMesh)
    {
        N = 0;
        delete mplPoint;
        delete mplNormal;
        mbHasMesh = false;
    }
}

void MapStruct::SetMeshEngine(MeshEngine *pMeshEngine)
{
    mpMeshEngine = pMeshEngine;
}

void MapStruct::SetRayTraceEngine(RayTraceEngine *pRayTraceEngine)
{
    mpRayTraceEngine = pRayTraceEngine;
}

void MapStruct::Swap(MapStruct *pMapStruct)
{
    using std::swap;
    swap(mplHeap, pMapStruct->mplHeap);
    swap(mplHeapPtr, pMapStruct->mplHeapPtr);
    swap(mplHashTable, pMapStruct->mplHashTable);
    swap(visibleTable, pMapStruct->visibleTable);
    swap(mplVoxelBlocks, pMapStruct->mplVoxelBlocks);
    swap(mplBucketMutex, pMapStruct->mplBucketMutex);
    swap(mpLinkedListHead, pMapStruct->mpLinkedListHead);
    swap(visibleBlockNum, pMapStruct->visibleBlockNum);

    swap(bucketSize, pMapStruct->bucketSize);
    swap(hashTableSize, pMapStruct->hashTableSize);
    swap(voxelBlockSize, pMapStruct->voxelBlockSize);
    swap(voxelSize, pMapStruct->voxelSize);
    swap(truncationDist, pMapStruct->truncationDist);

    swap(mbInHibernation, pMapStruct->mbInHibernation);
    swap(mbVertexBufferCreated, pMapStruct->mbVertexBufferCreated);
    swap(mnId, pMapStruct->mnId);
    swap(mK, pMapStruct->mK);
    swap(mbHasMesh, pMapStruct->mbHasMesh);
    swap(mplPoint, pMapStruct->mplPoint);
    swap(mplNormal, pMapStruct->mplNormal);
    swap(N, pMapStruct->N);
}

uint MapStruct::GetNumVisibleBlocks()
{
    uint nVisibleBlock = 0;
    SafeCall(cudaMemcpy(&nVisibleBlock, visibleBlockNum, sizeof(uint), cudaMemcpyDeviceToHost));
    return nVisibleBlock;
}

void MapStruct::ResetNumVisibleBlocks()
{
    SafeCall(cudaMemset(visibleBlockNum, 0, sizeof(uint)));
}

struct CreateBlockFunctor
{
    HashEntry *plDstEntry;
    HashEntry *plCurrEntry;
    int dstBucketSize;
    int currHashTableSize;
    int dstHashTableSize;
    int *plBucketMutex;
    Voxel *plVoxels;
    int *plHeapPtr;
    int *plHeap;
    int *pLinkedListPtr;
    float voxelSize;
    Sophus::SE3f T_src_dst;

    __device__ __forceinline__ void operator()() const;
};

__device__ __forceinline__ void CreateBlockFunctor::operator()() const
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= currHashTableSize)
        return;

    if (plCurrEntry[x].ptr == -1)
        return;

#pragma unroll
    for (int vid = 0; vid < BlockSize; ++vid)
    {
        Eigen::Vector3i localPos(threadIdx.x, threadIdx.y, vid);
        int localIdx = LocalPosToLocalIdx(localPos);
        Voxel &voxel_src = plVoxels[plCurrEntry[x].ptr + localIdx];
        if (voxel_src.wt == 0)
            continue;

        Eigen::Vector3i VoxelPos = BlockPosToVoxelPos(plCurrEntry[x].pos) + localPos;
        Eigen::Vector3f WorldPt = T_src_dst * VoxelPosToWorldPt(VoxelPos, voxelSize);
        VoxelPos = WorldPtToVoxelPos(WorldPt, voxelSize);
        Eigen::Vector3i blockPos = VoxelPosToBlockPos(VoxelPos);

        HashEntry *dstEntry = nullptr;
        bool found = findEntry(blockPos, dstEntry, plDstEntry, dstBucketSize);

        if (!found)
        {
            dstEntry = nullptr;
            while (!dstEntry)
                dstEntry = CreateNewBlock(blockPos, plHeap, plHeapPtr, plDstEntry,
                                          plBucketMutex, pLinkedListPtr,
                                          dstHashTableSize, dstBucketSize);
        }
    }
}

struct MapStructFusionFunctor
{
    HashEntry *plDstEntry;
    HashEntry *plCurrEntry;
    int currBucketSize;
    int currHashTableSize;
    int dstHashTableSize;
    int *plBucketMutex;
    Voxel *plVoxels;
    Voxel *plDstVoxels;
    float voxelSize;
    Sophus::SE3f T_dst_src;

    mutable uchar wtOld;
    mutable bool mbGetWt;

    __device__ __forceinline__ float ReadSDF(Eigen::Vector3i pt, bool &valid) const;
    __device__ __forceinline__ float InterpolateSDF(Eigen::Vector3f pt, bool &valid) const;
    __device__ __forceinline__ void operator()() const;
};

__device__ __forceinline__ float MapStructFusionFunctor::ReadSDF(Eigen::Vector3i pt, bool &valid) const
{
    Voxel *voxel = nullptr;
    findVoxel(pt, voxel, plCurrEntry, plVoxels, currBucketSize);
    if (voxel && voxel->wt != 0)
    {
        valid = true;
        if (mbGetWt)
        {
            wtOld = voxel->wt;
            mbGetWt = false;
        }
        return UnPackFloat(voxel->sdf);
    }
    else
    {
        valid = false;
        return 1.0f;
    }
}

__device__ __forceinline__ float MapStructFusionFunctor::InterpolateSDF(Eigen::Vector3f pt, bool &valid) const
{
    Eigen::Vector3f voxelPt = pt / voxelSize;
    Eigen::Vector3i voxelPos = WorldPtToVoxelPos(pt, voxelSize);
    float x = voxelPt(0) - voxelPos(0);
    float y = voxelPt(1) - voxelPos(1);
    float z = voxelPt(2) - voxelPos(2);
    // printf("%f,%f,%f,%d,%d,%d,%f,%f,%f\n", x, y, z, voxelPos(0), voxelPos(1), voxelPos(2), voxelPt(0), voxelPt(1), voxelPt(2));
    float sdf[4] = {1.f, 1.f, 1.f, 1.f};

    sdf[0] = ReadSDF(voxelPos, valid);
    if (!valid)
        return 1.f;
    sdf[1] = ReadSDF(voxelPos + Eigen::Vector3i(1, 0, 0), valid);
    if (!valid)
        return 1.f;

    sdf[2] = sdf[0] * (1 - x) + sdf[1] * x;

    sdf[0] = ReadSDF(voxelPos + Eigen::Vector3i(0, 1, 0), valid);
    if (!valid)
        return 1.f;
    sdf[1] = ReadSDF(voxelPos + Eigen::Vector3i(1, 1, 0), valid);
    if (!valid)
        return 1.f;

    sdf[3] = sdf[2] * (1 - y) + (sdf[0] * (1 - x) + sdf[1] * x) * y;

    sdf[0] = ReadSDF(voxelPos + Eigen::Vector3i(0, 0, 1), valid);
    if (!valid)
        return 1.f;
    sdf[1] = ReadSDF(voxelPos + Eigen::Vector3i(1, 0, 1), valid);
    if (!valid)
        return 1.f;

    sdf[2] = sdf[0] * (1 - x) + sdf[1] * x;

    sdf[0] = ReadSDF(voxelPos + Eigen::Vector3i(0, 1, 1), valid);
    if (!valid)
        return 1.f;
    sdf[1] = ReadSDF(voxelPos + Eigen::Vector3i(1, 1, 1), valid);
    if (!valid)
        return 1.f;

    return sdf[3] * (1 - z) + (sdf[2] * (1 - y) + (sdf[0] * (1 - x) + sdf[1] * x) * y) * z;
}

__device__ __forceinline__ void MapStructFusionFunctor::operator()() const
{
    int x = blockIdx.x;
    if (x >= dstHashTableSize)
        return;

    if (plDstEntry[x].ptr == -1)
        return;

#pragma unroll
    for (int vid = 0; vid < BlockSize; ++vid)
    {
        Eigen::Vector3i localPos(threadIdx.x, threadIdx.y, vid);
        int localIdx = LocalPosToLocalIdx(localPos);
        Voxel &dst = plDstVoxels[plDstEntry[x].ptr + localIdx];

        Eigen::Vector3i VoxelPos = BlockPosToVoxelPos(plDstEntry[x].pos) + localPos;
        Eigen::Vector3f WorldPt = T_dst_src * (VoxelPosToWorldPt(VoxelPos, voxelSize));
        bool valid = false;

        wtOld = 0;
        mbGetWt = true;
        float sdf = InterpolateSDF(WorldPt, valid);
        if (!valid)
            continue;

        float dstSdf = UnPackFloat(dst.sdf);
        dst.sdf = PackFloat((dstSdf * dst.wt + sdf * wtOld) / (dst.wt + wtOld));
        dst.wt = min(255, dst.wt + wtOld);
    }
}

void MapStruct::Fuse(MapStruct *pMapStruct)
{
    if (!pMapStruct || pMapStruct->Empty())
        return;

    if (Empty())
    {
        this->Swap(pMapStruct);
        return;
    }

    DeleteMesh();

    int nHashEntryCom = (hashTableSize + pMapStruct->hashTableSize);
    int nBucektCom = static_cast<int>(0.8 * nHashEntryCom);
    int nVoxelBlockCom = (voxelBlockSize + pMapStruct->voxelBlockSize);

    Reserve(nHashEntryCom, nBucektCom, nVoxelBlockCom);

    CreateBlockFunctor step1;
    step1.plDstEntry = mplHashTable;
    step1.plCurrEntry = pMapStruct->mplHashTable;
    step1.dstBucketSize = bucketSize;
    step1.currHashTableSize = pMapStruct->hashTableSize;
    step1.dstHashTableSize = hashTableSize;
    step1.plBucketMutex = mplBucketMutex;
    step1.plVoxels = pMapStruct->mplVoxelBlocks;
    step1.plHeapPtr = mplHeapPtr;
    step1.plHeap = mplHeap;
    step1.pLinkedListPtr = mpLinkedListHead;
    step1.T_src_dst = (GetPose().inverse() * pMapStruct->GetPose()).cast<float>();
    step1.voxelSize = voxelSize;

    dim3 block(8, 8);
    dim3 grid(pMapStruct->hashTableSize);
    callDeviceFunctor<<<grid, block>>>(step1);

    MapStructFusionFunctor step2;
    step2.plDstEntry = mplHashTable;
    step2.plCurrEntry = pMapStruct->mplHashTable;
    step2.currBucketSize = pMapStruct->bucketSize;
    step2.currHashTableSize = pMapStruct->hashTableSize;
    step2.dstHashTableSize = hashTableSize;
    step2.plBucketMutex = mplBucketMutex;
    step2.plVoxels = pMapStruct->mplVoxelBlocks;
    step2.plDstVoxels = mplVoxelBlocks;
    step2.T_dst_src = (pMapStruct->GetPose().inverse() * GetPose()).cast<float>();
    step2.voxelSize = voxelSize;

    grid = dim3(hashTableSize);
    callDeviceFunctor<<<grid, block>>>(step2);

    SafeCall(cudaDeviceSynchronize());

    pMapStruct->Release();
}

struct ResizeMapStructFunctor
{
    HashEntry *plDstEntry;
    HashEntry *plCurrEntry;
    int dstBucketSize;
    int currHashTableSize;
    int dstHashTableSize;
    int *plDstBucketMutex;
    Voxel *plCurrVoxels;
    Voxel *plDstVoxels;
    int *plDstHeapPtr;
    int *plDstHeap;
    int *pDstLinkedListPtr;

    __device__ __forceinline__ void operator()() const;
};

__device__ __forceinline__ void ResizeMapStructFunctor::operator()() const
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= currHashTableSize)
        return;

    if (plCurrEntry[x].ptr == -1)
        return;

    Voxel *src = &plCurrVoxels[plCurrEntry[x].ptr];
    Eigen::Vector3i blockPos = plCurrEntry[x].pos;

    HashEntry *pNewEntry = nullptr;
    while (!pNewEntry)
        pNewEntry = CreateNewBlock(
            blockPos, plDstHeap, plDstHeapPtr, plDstEntry,
            plDstBucketMutex, pDstLinkedListPtr,
            dstHashTableSize, dstBucketSize);

    Voxel *dst = &plDstVoxels[pNewEntry->ptr];
    memcpy(dst, src, sizeof(Voxel) * BlockSize3);
}

void MapStruct::Reserve(int hSize, int bSize, int vSize)
{
    if (hSize <= hashTableSize && bSize <= bucketSize && vSize <= voxelBlockSize)
        return;

    MapStruct *pNewMS = new MapStruct(mK);
    pNewMS->create(hSize, bSize, vSize, voxelSize, truncationDist);
    pNewMS->Reset();

    ResizeMapStructFunctor functor;
    functor.plDstEntry = pNewMS->mplHashTable;
    functor.plCurrEntry = mplHashTable;
    functor.dstBucketSize = pNewMS->bucketSize;
    functor.currHashTableSize = hashTableSize;
    functor.dstHashTableSize = pNewMS->hashTableSize;
    functor.plDstBucketMutex = pNewMS->mplBucketMutex;
    functor.plCurrVoxels = mplVoxelBlocks;
    functor.plDstVoxels = pNewMS->mplVoxelBlocks;
    functor.plDstHeapPtr = pNewMS->mplHeapPtr;
    functor.plDstHeap = pNewMS->mplHeap;
    functor.pDstLinkedListPtr = pNewMS->mpLinkedListHead;

    dim3 block(1024);
    dim3 grid(cv::divUp(hashTableSize, block.x));
    callDeviceFunctor<<<grid, block>>>(functor);

    Swap(pNewMS);
    pNewMS->Release();
    delete pNewMS;
}

struct CreateBlockLineTracingFunctor
{
    int *mplHeap;
    int *mplHeapPtr;
    HashEntry *mplHashTable;
    int *mplBucketMutex;
    int *mpLinkedListHead;
    int hashTableSize;
    int bucketSize;

    float voxelSize;
    float truncDistHalf;
    cv::cuda::PtrStepSz<float> depth;

    float invfx, invfy, cx, cy;
    float depthMin, depthMax;

    Sophus::SE3f T;

    // Allocate Blocks on the GPU memory
    __device__ __forceinline__ void allocateBlock(const Eigen::Vector3i &blockPos) const
    {
        HashEntry *pEntry = nullptr;
        while (!pEntry)
            pEntry = CreateNewBlock(blockPos, mplHeap, mplHeapPtr, mplHashTable,
                                    mplBucketMutex, mpLinkedListHead,
                                    hashTableSize, bucketSize);
    }

    // Allocate Blocks on the ray direction with a certain range
    // This is derived from Bresenham's line tracing algorithm
    // details see : https://en.m.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    __device__ __forceinline__ void operator()() const
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= depth.cols || y >= depth.rows)
            return;

        float dist = depth.ptr(y)[x];
        if (isnan(dist) || dist < depthMin || dist > depthMax)
            return;

        float distNear = max(depthMin, dist - truncDistHalf);
        float distFar = min(depthMax, dist + truncDistHalf);
        if (distNear >= distFar)
            return;

        Eigen::Vector3i blockStart = VoxelPosToBlockPos(WorldPtToVoxelPos(UnProjectWorld(x, y, distNear, invfx, invfy, cx, cy, T), voxelSize));
        Eigen::Vector3i blockEnd = VoxelPosToBlockPos(WorldPtToVoxelPos(UnProjectWorld(x, y, distFar, invfx, invfy, cx, cy, T), voxelSize));

        Eigen::Vector3i dir = blockEnd - blockStart;
        Eigen::Vector3i increment = Eigen::Vector3i(dir(0) < 0 ? -1 : 1, dir(1) < 0 ? -1 : 1, dir(2) < 0 ? -1 : 1);
        Eigen::Vector3i AbsIncre = Eigen::Vector3i(abs(dir(0)), abs(dir(1)), abs(dir(2)));
        Eigen::Vector3i IncreErr = Eigen::Vector3i(AbsIncre(0) << 1, AbsIncre(1) << 1, AbsIncre(2) << 1);

        int err1;
        int err2;

        if ((AbsIncre(0) >= AbsIncre(1)) && (AbsIncre(0) >= AbsIncre(2)))
        {
            err1 = IncreErr(1) - 1;
            err2 = IncreErr(2) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < AbsIncre(0); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(1) += increment(1);
                    err1 -= IncreErr(0);
                }

                if (err2 > 0)
                {
                    blockStart(2) += increment(2);
                    err2 -= IncreErr(0);
                }

                err1 += IncreErr(1);
                err2 += IncreErr(2);
                blockStart(0) += increment(0);
                allocateBlock(blockStart);
            }
        }
        else if ((AbsIncre(1) >= AbsIncre(0)) && (AbsIncre(1) >= AbsIncre(2)))
        {
            err1 = IncreErr(0) - 1;
            err2 = IncreErr(2) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < AbsIncre(1); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(0) += increment(0);
                    err1 -= IncreErr(1);
                }

                if (err2 > 0)
                {
                    blockStart(2) += increment(2);
                    err2 -= IncreErr(1);
                }

                err1 += IncreErr(0);
                err2 += IncreErr(2);
                blockStart(1) += increment(1);
                allocateBlock(blockStart);
            }
        }
        else
        {
            err1 = IncreErr(1) - 1;
            err2 = IncreErr(0) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < AbsIncre(2); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(1) += increment(1);
                    err1 -= IncreErr(2);
                }

                if (err2 > 0)
                {
                    blockStart(0) += increment(0);
                    err2 -= IncreErr(2);
                }

                err1 += IncreErr(1);
                err2 += IncreErr(0);
                blockStart(2) += increment(2);
                allocateBlock(blockStart);
            }
        }
    }
};

struct CheckEntryVisibilityFunctor
{
    HashEntry *mplHashTable;
    HashEntry *visibleEntry;
    uint *visibleEntryCount;
    Sophus::SE3f Tinv;

    int *mplHeap;
    int *mplHeapPtr;
    Voxel *mplVoxelBlocks;
    int cols, rows;
    float fx, fy;
    float cx, cy;
    float depthMin;
    float depthMax;
    float voxelSize;
    int hashTableSize;
    int voxelBlockSize;

    __device__ __forceinline__ void operator()() const
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        __shared__ bool needScan;

        if (threadIdx.x == 0)
            needScan = false;

        __syncthreads();

        uint increment = 0;
        if (idx < hashTableSize)
        {
            HashEntry *current = &mplHashTable[idx];
            if (current->ptr >= 0)
            {
                bool rval = CheckBlockVisibility(
                    current->pos,
                    Tinv,
                    voxelSize,
                    cols, rows,
                    fx, fy,
                    cx, cy,
                    depthMin,
                    depthMax);

                if (rval)
                {
                    needScan = true;
                    increment = 1;
                }
            }
        }

        __syncthreads();

        if (needScan)
        {
            auto offset = ParallelScan<1024>(increment, visibleEntryCount);
            if (offset >= 0 && offset < hashTableSize && idx < hashTableSize)
                visibleEntry[offset] = mplHashTable[idx];
        }
    }
};

struct DepthFusionFunctor
{

    Voxel *listBlock;
    HashEntry *visible_blocks;

    Sophus::SE3f Tinv;
    float fx, fy;
    float cx, cy;
    float depthMin;
    float depthMax;

    float truncationDist;
    int hashTableSize;
    float voxelSize;
    uint count_visible_block;

    cv::cuda::PtrStepSz<float> depth;

    __device__ __forceinline__ void operator()() const
    {
        if (blockIdx.x >= hashTableSize || blockIdx.x >= count_visible_block)
            return;

        HashEntry &current = visible_blocks[blockIdx.x];
        if (current.ptr == -1)
            return;

        Eigen::Vector3i voxelPos = BlockPosToVoxelPos(current.pos);

#pragma unroll
        for (int blockIdxZ = 0; blockIdxZ < 8; ++blockIdxZ)
        {
            Eigen::Vector3i localPos = Eigen::Vector3i(threadIdx.x, threadIdx.y, blockIdxZ);
            Eigen::Vector3f pt = Tinv * VoxelPosToWorldPt(voxelPos + localPos, voxelSize);

            int u = __float2int_rd(fx * pt(0) / pt(2) + cx + 0.5);
            int v = __float2int_rd(fy * pt(1) / pt(2) + cy + 0.5);
            if (u < 0 || v < 0 || u > depth.cols - 1 || v > depth.rows - 1)
                continue;

            float dist = depth.ptr(v)[u];
            if (isnan(dist) || dist > depthMax || dist < depthMin)
                continue;

            float sdf = dist - pt(2);
            if (sdf < -truncationDist)
                continue;

            sdf = fmin(1.0f, sdf / truncationDist);
            const int localIdx = LocalPosToLocalIdx(localPos);
            Voxel &voxel = listBlock[current.ptr + localIdx];

            auto oldSDF = UnPackFloat(voxel.sdf);
            auto oldWT = voxel.wt;

            if (oldWT == 0)
            {
                voxel.sdf = PackFloat(sdf);
                voxel.wt = 1;
                continue;
            }

            voxel.sdf = PackFloat((oldSDF * oldWT + sdf * 1) / (oldWT + 1));
            voxel.wt = min(255, oldWT + 1);
        }
    }
};

uint MapStruct::CheckNumVisibleBlocks(int cols, int rows, const Sophus::SE3d &Tcm)
{
    ResetNumVisibleBlocks();

    float fx = mK(0, 0);
    float fy = mK(1, 1);
    float cx = mK(0, 2);
    float cy = mK(1, 2);

    CheckEntryVisibilityFunctor functor;
    functor.mplHashTable = mplHashTable;
    functor.mplVoxelBlocks = mplVoxelBlocks;
    functor.visibleEntry = visibleTable;
    functor.visibleEntryCount = visibleBlockNum;
    functor.mplHeap = mplHeap;
    functor.mplHeapPtr = mplHeapPtr;
    functor.voxelBlockSize = voxelBlockSize;
    functor.Tinv = Tcm.inverse().cast<float>();
    functor.cols = cols;
    functor.rows = rows;
    functor.fx = fx;
    functor.fy = fy;
    functor.cx = cx;
    functor.cy = cy;
    functor.depthMin = 0.1f;
    functor.depthMax = 3.0f;
    functor.voxelSize = voxelSize;
    functor.hashTableSize = hashTableSize;

    dim3 block = dim3(1024);
    dim3 grid = dim3(cv::divUp(hashTableSize, block.x));

    callDeviceFunctor<<<grid, block>>>(functor);

    return GetNumVisibleBlocks();
}

void MapStruct::Fuse(cv::cuda::GpuMat depth, const Sophus::SE3d &Tcm)
{
    float fx = mK(0, 0);
    float fy = mK(1, 1);
    float cx = mK(0, 2);
    float cy = mK(1, 2);
    float invfx = 1.0 / mK(0, 0);
    float invfy = 1.0 / mK(1, 1);

    const int cols = depth.cols;
    const int rows = depth.rows;

    CreateBlockLineTracingFunctor step1;
    step1.mplHeap = mplHeap;
    step1.mplHeapPtr = mplHeapPtr;
    step1.mplHashTable = mplHashTable;
    step1.mplBucketMutex = mplBucketMutex;
    step1.mpLinkedListHead = mpLinkedListHead;
    step1.hashTableSize = hashTableSize;
    step1.bucketSize = bucketSize;
    step1.voxelSize = voxelSize;
    step1.truncDistHalf = truncationDist * 0.5;
    step1.depth = depth;
    step1.invfx = invfx;
    step1.invfy = invfy;
    step1.cx = cx;
    step1.cy = cy;
    step1.depthMin = 0.1f;
    step1.depthMax = 3.0f;
    step1.T = Tcm.cast<float>();

    dim3 block(8, 8);
    dim3 grid(cv::divUp(cols, block.x), cv::divUp(rows, block.y));
    callDeviceFunctor<<<grid, block>>>(step1);

    uint nVisibleBlock = CheckNumVisibleBlocks(cols, rows, Tcm);

    if (nVisibleBlock == 0)
        return;

    DepthFusionFunctor step3;
    step3.listBlock = mplVoxelBlocks;
    step3.visible_blocks = visibleTable;
    step3.Tinv = Tcm.inverse().cast<float>();
    step3.fx = fx;
    step3.fy = fy;
    step3.cx = cx;
    step3.cy = cy;
    step3.depthMin = 0.1f;
    step3.depthMax = 3.0f;
    step3.truncationDist = truncationDist;
    step3.hashTableSize = hashTableSize;
    step3.voxelSize = voxelSize;
    step3.count_visible_block = nVisibleBlock;
    step3.depth = depth;

    block = dim3(8, 8);
    grid = dim3(nVisibleBlock);
    callDeviceFunctor<<<grid, block>>>(step3);
}

void MapStruct::SaveToFile(std::string &strFileName)
{
}

void MapStruct::ReadFromFile(std::string &strFileName)
{
}

void MapStruct::Hibernate()
{
    if (mbInHibernation || Empty())
        return;

    mpLinkedListHeadHib = new int[1];
    mplHeapPtrHib = new int[1];
    mplBucketMutexHib = new int[bucketSize];
    mplHeapHib = new int[voxelBlockSize];
    mplHashTableHib = new HashEntry[hashTableSize];
    mplVoxelBlocksHib = new Voxel[voxelBlockSize * BlockSize3];

    SafeCall(cudaMemcpy(mpLinkedListHeadHib, mpLinkedListHead, sizeof(int), cudaMemcpyDeviceToHost));
    SafeCall(cudaMemcpy(mplHeapPtrHib, mplHeapPtr, sizeof(int), cudaMemcpyDeviceToHost));
    SafeCall(cudaMemcpy(mplBucketMutexHib, mplBucketMutex, sizeof(int) * bucketSize, cudaMemcpyDeviceToHost));
    SafeCall(cudaMemcpy(mplHeapHib, mplHeap, sizeof(int) * voxelBlockSize, cudaMemcpyDeviceToHost));
    SafeCall(cudaMemcpy(mplHashTableHib, mplHashTable, sizeof(HashEntry) * hashTableSize, cudaMemcpyDeviceToHost));
    SafeCall(cudaMemcpy(mplVoxelBlocksHib, mplVoxelBlocks, sizeof(Voxel) * voxelBlockSize * BlockSize3, cudaMemcpyDeviceToHost));

    mbInHibernation = true;

    SafeCall(cudaFree((void *)mplHeap));
    SafeCall(cudaFree((void *)mplHeapPtr));
    SafeCall(cudaFree((void *)mplHashTable));
    SafeCall(cudaFree((void *)mplBucketMutex));
    SafeCall(cudaFree((void *)mpLinkedListHead));
    SafeCall(cudaFree((void *)mplVoxelBlocks));
    SafeCall(cudaFree((void *)visibleBlockNum));
    SafeCall(cudaFree((void *)visibleTable));
}

void MapStruct::ReActivate()
{
    if (!mbInHibernation || Empty())
        return;

    SafeCall(cudaMalloc((void **)&mpLinkedListHead, sizeof(int)));
    SafeCall(cudaMalloc((void **)&mplHeapPtr, sizeof(int)));
    SafeCall(cudaMalloc((void **)&visibleBlockNum, sizeof(uint)));
    SafeCall(cudaMalloc((void **)&mplBucketMutex, sizeof(int) * bucketSize));
    SafeCall(cudaMalloc((void **)&mplHeap, sizeof(int) * voxelBlockSize));
    SafeCall(cudaMalloc((void **)&mplHashTable, sizeof(HashEntry) * hashTableSize));
    SafeCall(cudaMalloc((void **)&visibleTable, sizeof(HashEntry) * hashTableSize));
    SafeCall(cudaMalloc((void **)&mplVoxelBlocks, sizeof(Voxel) * voxelBlockSize * BlockSize3));

    SafeCall(cudaMemcpy(mpLinkedListHead, mpLinkedListHeadHib, sizeof(int), cudaMemcpyHostToDevice));
    SafeCall(cudaMemcpy(mplHeapPtr, mplHeapPtrHib, sizeof(int), cudaMemcpyHostToDevice));
    SafeCall(cudaMemcpy(mplBucketMutex, mplBucketMutexHib, sizeof(int) * bucketSize, cudaMemcpyHostToDevice));
    SafeCall(cudaMemcpy(mplHeap, mplHeapHib, sizeof(int) * voxelBlockSize, cudaMemcpyHostToDevice));
    SafeCall(cudaMemcpy(mplHashTable, mplHashTableHib, sizeof(HashEntry) * hashTableSize, cudaMemcpyHostToDevice));
    SafeCall(cudaMemcpy(mplVoxelBlocks, mplVoxelBlocksHib, sizeof(Voxel) * voxelBlockSize * BlockSize3, cudaMemcpyHostToDevice));

    mbInHibernation = false;

    delete mpLinkedListHeadHib;
    delete mplHeapPtrHib;
    delete mplBucketMutexHib;
    delete mplHeapHib;
    delete mplHashTableHib;
    delete mplVoxelBlocksHib;
}

uint MapStruct::GetVisibleBlocks()
{
    uint temp = 0;
    SafeCall(cudaMemcpy(&temp, visibleBlockNum, sizeof(uint), cudaMemcpyDeviceToHost));
    return temp;
}

void MapStruct::ResetVisibleBlocks()
{
    SafeCall(cudaMemset(visibleBlockNum, 0, sizeof(uint)));
}

void MapStruct::RayTrace(const Sophus::SE3d &Tcm)
{
    if (mpRayTraceEngine)
    {
        mpRayTraceEngine->RayTrace(this, Tcm);
    }
}

cv::cuda::GpuMat MapStruct::GetRayTracingResult()
{
    return mpRayTraceEngine->GetVMap();
}

cv::cuda::GpuMat MapStruct::GetRayTracingResultDepth()
{
    auto vmap = mpRayTraceEngine->GetVMap();
    cv::cuda::GpuMat depth;
    VMapToDepth(vmap, depth);
    return depth;
}

void MapStruct::SetActiveFlag(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexActive);
    mbActive = flag;
}

bool MapStruct::isActive()
{
    std::unique_lock<std::mutex> lock(mMutexActive);
    return mbActive;
}