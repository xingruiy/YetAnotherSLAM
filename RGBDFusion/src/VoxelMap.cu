#include "VoxelMap.h"
#include "CudaUtils.h"
#include "ParallelScan.h"
#include "VoxelStructUtils.h"

#define HASHENTRY_IN_BYTE 0.00002
#define VOXEL_BLOCK_IN_BYTE 0.000003

long unsigned int MapStruct::nNextId = 0;

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
    : mFootPrintInMB(0), mbInHibernation(false), mbActive(false),
      mbHasMesh(false), mpMeshEngine(NULL), mplHeap(NULL),
      mplHeapPtr(NULL), mplBucketMutex(NULL), mplHashTable(NULL),
      mplVoxelBlocks(NULL), mpLinkedListHead(NULL), mK(K),
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
    SafeCall(cudaFree((void *)mplHeap));
    SafeCall(cudaFree((void *)mplHeapPtr));
    SafeCall(cudaFree((void *)mplHashTable));
    SafeCall(cudaFree((void *)mplBucketMutex));
    SafeCall(cudaFree((void *)mpLinkedListHead));
    SafeCall(cudaFree((void *)mplVoxelBlocks));
    SafeCall(cudaFree((void *)visibleBlockNum));
    SafeCall(cudaFree((void *)visibleTable));

    if (mbHasMesh && N > 0)
    {
        N = 0;
        free(mplPoint);
        free(mplNormal);
    }

    mplHeap = NULL;
    mplHeapPtr = NULL;
    mplHashTable = NULL;
    visibleTable = NULL;
    mplVoxelBlocks = NULL;
    mplBucketMutex = NULL;
    visibleBlockNum = NULL;
    mpLinkedListHead = NULL;
    mbInHibernation = false;
    mFootPrintInMB = 0;
}

bool MapStruct::empty()
{
    return bucketSize == 0;
}

void MapStruct::GenerateMesh()
{
    if (!mbHasMesh && mpMeshEngine && !mbInHibernation)
    {
        mpMeshEngine->Meshify(this);
        SafeCall(cudaDeviceSynchronize());
        SafeCall(cudaGetLastError());

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
    }
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

struct FuseMapStruct_functor
{
    HashEntry *plDstEntry;
    HashEntry *plCurrEntry;
    int dstBucketSize;
    int currHashTableSize;
    int dstHashTableSize;
    int *plBucketMutex;
    Voxel *plVoxels;
    Voxel *plDstVoxels;
    int *plHeapPtr;
    int *plHeap;
    int *pLinkedListPtr;

    __device__ __forceinline__ int createNewBlock() const;
    __device__ __forceinline__ void move(Voxel *src, Voxel *dst) const;
    __device__ __forceinline__ void fuse(Voxel *src, Voxel *dst) const;
    __device__ __forceinline__ void operator()() const;
};

__device__ __forceinline__ int FuseMapStruct_functor::createNewBlock() const
{
    int old = atomicSub(plHeapPtr, 1);
    if (old > 0)
    {
        return plHeap[old];
    }
    else
    {
        atomicAdd(plHeapPtr, 1);
        return -1;
    }
}

__device__ __forceinline__ void FuseMapStruct_functor::move(Voxel *src, Voxel *dst) const
{
    memcpy(dst, src, sizeof(Voxel) * BlockSize3);
}

__device__ __forceinline__ void FuseMapStruct_functor::fuse(Voxel *src, Voxel *dst) const
{
    for (int i = 0; i < BlockSize3; ++i)
    {
        float dstSdf = UnPackFloat(dst[i].sdf);
        float srcSdf = UnPackFloat(src[i].sdf);
        dst[i].sdf = PackFloat(dstSdf * dst[i].wt + srcSdf * src[i].wt);
        dst[i].wt = min(255, dst[i].wt + src[i].wt);
    }
}

__device__ __forceinline__ void FuseMapStruct_functor::operator()() const
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= currHashTableSize)
        return;

    if (plCurrEntry[x].ptr == -1)
        return;

    Eigen::Vector3i pos = plCurrEntry[x].pos;
    Voxel *voxels = &plVoxels[plCurrEntry[x].ptr];
    uint hashIdx = hash(pos, dstBucketSize);
    int *mutex = &plBucketMutex[hashIdx];
    HashEntry *current = &plDstEntry[hashIdx];
    HashEntry *empty = nullptr;

    if (current->pos == pos && current->ptr != -1)
    {
        fuse(voxels, &plDstVoxels[current->ptr]);
        return;
    }
    else if (current->ptr == -1)
        empty = current;

    // search through the linked list
    while (current->offset >= 0)
    {
        hashIdx = dstBucketSize + current->offset - 1;
        current = &plDstEntry[hashIdx];
        if (current->pos == pos && current->ptr != -1)
        {
            fuse(voxels, &plDstVoxels[current->ptr]);
            return;
        }
        else if (!empty && current->ptr == -1)
            empty = current;
    }

    // if no existing block is found we create a new one
    if (empty)
    {
        if (LockBucket(mutex))
        {
            int new_ptr = createNewBlock();
            empty->pos = pos;
            empty->ptr = new_ptr;
            move(voxels, &plDstVoxels[new_ptr]);
            UnLockBucket(mutex);
        }
    }
    else
    {
        // we allocate a new one from the linked list
        if (LockBucket(mutex))
        {
            int offset = atomicAdd(pLinkedListPtr, 1);
            if ((offset + dstBucketSize) < dstHashTableSize)
            {
                empty = &plDstEntry[dstBucketSize + offset - 1];
                int new_ptr = createNewBlock();
                empty->ptr = new_ptr;
                empty->pos = pos;
                empty->offset = -1;
                current->offset = offset;
            }
            else
            {
                atomicSub(pLinkedListPtr, 1);
            }

            UnLockBucket(mutex);
        }
    }
}

void MapStruct::Fuse(MapStruct *pMapStruct)
{
    if (!pMapStruct || pMapStruct->empty())
        return;

    if (this->empty())
    {
        this->Swap(pMapStruct);
        return;
    }

    int nHashEntryCom = hashTableSize + pMapStruct->hashTableSize;
    int nBucektCom = static_cast<int>(0.8 * nHashEntryCom);
    int nVoxelBlockCom = voxelBlockSize + pMapStruct->voxelBlockSize;

    MapStruct *pNewMS = new MapStruct(mK);
    pNewMS->create(nHashEntryCom, nBucektCom,
                   nVoxelBlockCom, voxelSize,
                   truncationDist);

    FuseMapStruct_functor functor;
    functor.plDstEntry = pNewMS->mplHashTable;
    functor.plCurrEntry = mplHashTable;
    functor.dstBucketSize = pNewMS->bucketSize;
    functor.currHashTableSize = hashTableSize;
    functor.dstHashTableSize = pNewMS->hashTableSize;
    functor.plBucketMutex = pNewMS->mplBucketMutex;
    functor.plVoxels = mplVoxelBlocks;
    functor.plDstVoxels = pNewMS->mplVoxelBlocks;
    functor.plHeapPtr = pNewMS->mplHeapPtr;
    functor.plHeap = pNewMS->mplHeap;
    functor.pLinkedListPtr = pNewMS->mpLinkedListHead;

    dim3 block(1024);
    dim3 grid(cv::divUp(hashTableSize, block.x));
    callDeviceFunctor<<<grid, block>>>(functor);

    functor.plDstEntry = pNewMS->mplHashTable;
    functor.plCurrEntry = pMapStruct->mplHashTable;
    functor.dstBucketSize = pNewMS->bucketSize;
    functor.currHashTableSize = pMapStruct->hashTableSize;
    functor.dstHashTableSize = pNewMS->hashTableSize;
    functor.plBucketMutex = pNewMS->mplBucketMutex;
    functor.plVoxels = pMapStruct->mplVoxelBlocks;
    functor.plDstVoxels = pNewMS->mplVoxelBlocks;
    functor.plHeapPtr = pNewMS->mplHeapPtr;
    functor.plHeap = pNewMS->mplHeap;
    functor.pLinkedListPtr = pNewMS->mpLinkedListHead;

    block = dim3(1024);
    grid = dim3(cv::divUp(pMapStruct->hashTableSize, block.x));
    callDeviceFunctor<<<grid, block>>>(functor);

    this->Release();
    pMapStruct->Release();
    Swap(pNewMS);
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

    __device__ __forceinline__ void allocateBlock(const Eigen::Vector3i &blockPos) const
    {
        CreateNewBlock(blockPos, mplHeap, mplHeapPtr, mplHashTable, mplBucketMutex, mpLinkedListHead, hashTableSize, bucketSize);
    }

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
        Eigen::Vector3i absIncrement = Eigen::Vector3i(abs(dir(0)), abs(dir(1)), abs(dir(2)));
        Eigen::Vector3i incrementErr = Eigen::Vector3i(absIncrement(0) << 1, absIncrement(1) << 1, absIncrement(2) << 1);

        int err1;
        int err2;

        // Bresenham's line algorithm
        // details see : https://en.m.wikipedia.org/wiki/Bresenham%27s_line_algorithm
        if ((absIncrement(0) >= absIncrement(1)) && (absIncrement(0) >= absIncrement(2)))
        {
            err1 = incrementErr(1) - 1;
            err2 = incrementErr(2) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < absIncrement(0); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(1) += increment(1);
                    err1 -= incrementErr(0);
                }

                if (err2 > 0)
                {
                    blockStart(2) += increment(2);
                    err2 -= incrementErr(0);
                }

                err1 += incrementErr(1);
                err2 += incrementErr(2);
                blockStart(0) += increment(0);
                allocateBlock(blockStart);
            }
        }
        else if ((absIncrement(1) >= absIncrement(0)) && (absIncrement(1) >= absIncrement(2)))
        {
            err1 = incrementErr(0) - 1;
            err2 = incrementErr(2) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < absIncrement(1); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(0) += increment(0);
                    err1 -= incrementErr(1);
                }

                if (err2 > 0)
                {
                    blockStart(2) += increment(2);
                    err2 -= incrementErr(1);
                }

                err1 += incrementErr(0);
                err2 += incrementErr(2);
                blockStart(1) += increment(1);
                allocateBlock(blockStart);
            }
        }
        else
        {
            err1 = incrementErr(1) - 1;
            err2 = incrementErr(0) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < absIncrement(2); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(1) += increment(1);
                    err1 -= incrementErr(2);
                }

                if (err2 > 0)
                {
                    blockStart(0) += increment(0);
                    err2 -= incrementErr(2);
                }

                err1 += incrementErr(1);
                err2 += incrementErr(0);
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
    if (mbInHibernation || empty())
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
    if (!mbInHibernation || empty())
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