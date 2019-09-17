#include "utils/numType.h"
#include "utils/prefixSum.h"
#include "localMapper/denseMap.h"

__device__ __forceinline__ Vec2f project(
    const Vec3f &pt, const float &fx, const float &fy,
    const float &cx, const float &cy)
{
    Vec2f ptWarped;
    ptWarped(0) = fx * ptWarped(0) / ptWarped(2) + cx;
    ptWarped(1) = fy * ptWarped(1) / ptWarped(2) + cy;
    return ptWarped;
}

__device__ __forceinline__ Vec3f unproject(
    const int &x, const int &y, const float &z,
    const float &invfx, const float &invfy,
    const float &cx, const float &cy)
{
    Vec3f pt;
    pt(0) = z * (x - cx) * invfx;
    pt(1) = z * (y - cy) * invfy;
    pt(2) = z;
    return pt;
}

__device__ __forceinline__ Vec3f unprojectWorld(
    const int &x, const int &y, const float &z,
    const float &invfx, const float &invfy,
    const float &cx, const float &cy, const SE3f &T)
{
    return T * unproject(x, y, z, invfx, invfy, cx, cy);
}

__device__ __forceinline__ bool checkVertexVisibility(
    const Vec3f &pt, const SE3f &TInv,
    const int &cols, const int &rows,
    const float &fx, const float &fy,
    const float &cx, const float &cy,
    const float &depthMin, const float &depthMax)
{
    auto ptFrame = TInv * pt;
    auto ptWarped = project(ptFrame, fx, fy, cx, cy);

    return ptWarped(0) >= 0 && ptWarped(1) >= 0 &&
           ptWarped(0) < cols && ptWarped(1) < rows &&
           ptFrame(2) >= depthMin && ptFrame(2) <= depthMax;
}

__device__ __forceinline__ bool checkBlockVisibility(
    const Vec3f &blockPos, const SE3f &TInv,
    const int &cols, const int &rows,
    const float &fx, const float &fy,
    const float &cx, const float &cy,
    const float &depthMin, const float &depthMax,
    const float &voxelSize)
{

    float scale = voxelSize * BlockSize;
#pragma unroll
    for (int corner = 0; corner < 8; ++corner)
    {
        auto tmp = blockPos;
        tmp(0) += (corner & 1) ? 1 : 0;
        tmp(1) += (corner & 2) ? 1 : 0;
        tmp(2) += (corner & 4) ? 1 : 0;

        if (checkVertexVisibility(tmp * scale, TInv, cols, rows, fx, fy, cx, cy, depthMin, depthMax))
            return true;
    }

    return false;
}

struct CreateVoxelBlockFunctor
{
    SE3f T;
    float invfx, invfy;
    float cx, cy;

    float truncDistHalf;
    float voxelSizeInv;

    float depthMin, depthMax;
    int cols, rows;
    int numEntry, numBucket;
    HashEntry *hashTable;
    int *bucketMutex;
    int *heap;
    int *heapPtr;
    int *excessPtr;
    cv::cuda::PtrStep<float> depth;

    __device__ __forceinline__ void operator()() const
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= cols && y >= rows)
            return;

        float dist = depth.ptr(y)[x];
        if (dist < FLT_EPSILON || dist < depthMin || dist > depthMax)
            return;

        float distNear = fmax(depthMin, dist - truncDistHalf);
        float distFar = fmin(depthMax, dist + truncDistHalf);
        if (distNear >= distFar)
            return;

        Vec3f ptNear = unprojectWorld(x, y, distNear, invfx, invfy, cx, cy, T) * voxelSizeInv;
        Vec3f ptFar = unprojectWorld(x, y, distFar, invfx, invfy, cx, cy, T) * voxelSizeInv;
        Vec3f dir = ptFar - ptNear;

        float length = dir.norm();
        int nSteps = (int)ceil(2.0 * length);
        dir = dir / (float)(nSteps - 1);

        for (int i = 0; i < nSteps; ++i)
        {
            auto blockPos = voxelPosToBlockPos(ptNear.cast<int>());
            createBlock(hashTable, bucketMutex, heap, heapPtr, excessPtr, numEntry, numBucket, blockPos);
            ptNear += dir;
        }
    }
};

struct CheckEntryVisibilityFunctor
{
    SE3f TInv;
    int cols, rows;
    float fx, fy, cx, cy;
    float depthMin, depthMax;
    float voxelSize;
    int numEntry;
    HashEntry *hashTable;
    HashEntry *visibleEntry;

    uint *numVisibleEntry;

    __device__ __forceinline__ void operator()() const
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        __shared__ bool needScan;
        if (threadIdx.x == 0)
            needScan = false;

        __syncthreads();

        uint increment = 0;
        if (idx < numEntry)
        {
            HashEntry &entry = hashTable[idx];
            if (entry.ptr != -1)
            {
                if (checkBlockVisibility(
                        entry.pos.cast<float>(), TInv,
                        cols, rows,
                        fx, fy,
                        cx, cy,
                        depthMin, depthMax,
                        voxelSize))
                {
                    needScan = true;
                    increment = 1;
                }
            }
        }

        __syncthreads();

        if (needScan)
        {
            int offset = computeOffset<1024>(increment, numVisibleEntry);
            if (offset != -1 && offset < numEntry && idx < numEntry)
                visibleEntry[offset] = hashTable[idx];
        }
    }
};

struct DepthFusionFunctor
{
    uint numEntry;
    uint numVisibleEntry;
    cv::cuda::PtrStep<float> depth;
    HashEntry *visibleEntry;
    Voxel *voxels;
    SE3f TInv;

    int cols, rows;
    float fx, fy, cx, cy;

    float depthMax, depthMin;
    float truncDist;
    float voxelSize;

    __device__ __forceinline__ void integrateColor() const
    {
        if (blockIdx.x >= numEntry || blockIdx.x >= numVisibleEntry)
            return;

        HashEntry &entry = visibleEntry[blockIdx.x];
        if (entry.ptr == -1)
            return;

        auto blockPos = entry.pos * BlockSize;

#pragma unroll
        for (int blockIdxZ = 0; blockIdxZ < 8; ++blockIdxZ)
        {
            auto localPos = Vec3i(threadIdx.x, threadIdx.y, blockIdxZ);
            auto pt = TInv * (blockPos + localPos).cast<float>() * voxelSize;
            int u = __float2int_rd(fx * pt(0) / pt(2) + cx + 0.5f);
            int v = __float2int_rd(fy * pt(1) / pt(2) + cy + 0.5f);
            if (u < 0 || v < 0 || u >= cols || v >= rows)
                continue;

            float dist = depth.ptr(v)[u];
            if (dist < FLT_EPSILON || dist > depthMax || dist < depthMin)
                continue;

            float newSDF = dist - pt(2);
            if (newSDF < -truncDist)
                continue;

            newSDF = fmin(1.0f, newSDF / truncDist);
            int localIdx = localPosToLocalIdx(localPos);
            Voxel &curr = voxels[entry.ptr + localIdx];

            float oldSDF = unpackFloat(curr.sdf);
            uchar oldWT = curr.wt;

            if (oldWT == 0)
            {
                curr.sdf = packFloat(newSDF);
                curr.wt = 1;
                continue;
            }

            curr.sdf = packFloat((oldWT * oldSDF + newSDF) / (oldWT + 1));
            curr.wt = min(255, oldWT + 1);
        }
    }
};