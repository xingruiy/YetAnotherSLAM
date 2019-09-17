#pragma once
#include "utils/numType.h"
#include "utils/prefixSum.h"
#include "localMapper/denseMap.h"

#define RenderingBlockSizeX 16
#define RenderingBlockSizeY 16
#define RenderingBlockSubSample 8
#define MaxNumRenderingBlock 100000

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

    __device__ __forceinline__ void operator()() const
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

#define RenderingBlockSizeX 16
#define RenderingBlockSizeY 16
#define RenderingBlockSubSample 8

// compare val with the old value stored in *add
// and write the bigger one to *add
__device__ __forceinline__ void atomicMax(float *add, float val)
{
    int *address_as_i = (int *)add;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// compare val with the old value stored in *add
// and write the smaller one to *add
__device__ __forceinline__ void atomicMin(float *add, float val)
{
    int *address_as_i = (int *)add;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

struct ProjectBlockFunctor
{
    SE3f TInv;
    float fx, fy, cx, cy;

    float scale;
    float depthMin, depthMax;

    uint *numRenderingBlock;
    uint numVisibleEntry;

    HashEntry *visibleEntry;
    RenderingBlock *renderingBlock;
    mutable cv::cuda::PtrStepSz<float> zRangeX;
    mutable cv::cuda::PtrStep<float> zRangeY;

    __device__ __forceinline__ bool projectBlock(const Vec3i &blockPos, RenderingBlock &block) const
    {
        block.upperLeft = Vec2s(zRangeX.cols, zRangeX.rows);
        block.lowerRight = Vec2s(-1, -1);
        block.zrange = Vec2f(depthMax, depthMin);

#pragma unroll
        for (int corner = 0; corner < 8; ++corner)
        {
            Vec3f tmp = blockPos.cast<float>();
            tmp(0) += (corner & 1) ? 1 : 0;
            tmp(1) += (corner & 2) ? 1 : 0;
            tmp(2) += (corner & 4) ? 1 : 0;

            Vec3f ptTransformed = TInv * tmp * scale;
            Vec2f ptWarped = project(ptTransformed, fx, fy, cx, cy) / RenderingBlockSubSample;

            if (block.upperLeft(0) > std::floor(ptWarped(0)))
                block.upperLeft(0) = (int)std::floor(ptWarped(0));
            if (block.lowerRight(0) < ceil(ptWarped(0)))
                block.lowerRight(0) = (int)ceil(ptWarped(0));
            if (block.upperLeft(1) > std::floor(ptWarped(1)))
                block.upperLeft(1) = (int)std::floor(ptWarped(1));
            if (block.lowerRight(1) < ceil(ptWarped(1)))
                block.lowerRight(1) = (int)ceil(ptWarped(1));
            if (block.zrange(0) > ptTransformed(2))
                block.zrange(0) = ptTransformed(2);
            if (block.zrange(1) < ptTransformed(2))
                block.zrange(1) = ptTransformed(2);
        }

        if (block.upperLeft(0) < 0)
            block.upperLeft(0) = 0;
        if (block.upperLeft(1) < 0)
            block.upperLeft(1) = 0;
        if (block.lowerRight(0) >= zRangeX.cols)
            block.lowerRight(0) = zRangeX.cols - 1;
        if (block.lowerRight(1) >= zRangeX.rows)
            block.lowerRight(1) = zRangeX.rows - 1;
        if (block.upperLeft(0) > block.lowerRight(0))
            return false;
        if (block.upperLeft(1) > block.lowerRight(1))
            return false;
        if (block.zrange(0) < depthMin)
            block.zrange(0) = depthMin;
        if (block.zrange(1) < depthMin)
            return false;
        return true;
    }

    __device__ __forceinline__ void splitRenderingBlock(
        int offset, const RenderingBlock &block,
        const int &nx, const int &ny) const
    {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                if (offset < MaxNumRenderingBlock)
                {
                    RenderingBlock &b(renderingBlock[offset++]);
                    b.upperLeft(0) = block.upperLeft(0) + x * RenderingBlockSizeX;
                    b.upperLeft(1) = block.upperLeft(1) + y * RenderingBlockSizeY;
                    b.lowerRight(0) = block.upperLeft(0) + (x + 1) * RenderingBlockSizeX;
                    b.lowerRight(1) = block.upperLeft(1) + (y + 1) * RenderingBlockSizeY;

                    if (b.lowerRight(0) > block.lowerRight(0))
                        b.lowerRight(0) = block.lowerRight(0);

                    if (b.lowerRight(1) > block.lowerRight(1))
                        b.lowerRight(1) = block.lowerRight(1);

                    b.zrange = block.zrange;
                }
    }

    __device__ __forceinline__ void operator()() const
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        bool valid = false;
        uint requiredBlocks = 0;
        RenderingBlock block;
        int nx, ny;

        HashEntry &curr = visibleEntry[idx];
        if (idx < numVisibleEntry && curr.ptr != -1)
        {
            valid = projectBlock(curr.pos, block);
            if (valid)
            {
                float dx = (float)block.lowerRight(0) - block.upperLeft(0) + 1;
                float dy = (float)block.lowerRight(1) - block.upperLeft(1) + 1;
                nx = __float2int_ru(dx / RenderingBlockSizeX);
                ny = __float2int_ru(dy / RenderingBlockSizeY);
                requiredBlocks = nx * ny;
                uint totalBlocks = *numRenderingBlock + requiredBlocks;
                if (totalBlocks >= MaxNumRenderingBlock)
                    requiredBlocks = 0;
            }
        }

        int offset = computeOffset<1024>(requiredBlocks, numRenderingBlock);
        if (valid && offset != -1 && (offset + requiredBlocks) < MaxNumRenderingBlock)
            splitRenderingBlock(offset, block, nx, ny);
    }
};

struct DepthPredictionFunctor
{
    uint numRenderingBlock;
    RenderingBlock *renderingBlock;

    mutable cv::cuda::PtrStepSz<float> zRangeX;
    mutable cv::cuda::PtrStep<float> zRangeY;

    __device__ __forceinline__ void operator()() const
    {
        int x = threadIdx.x;
        int y = threadIdx.y;

        int block = blockIdx.x * 4 + blockIdx.y;
        if (block >= numRenderingBlock)
            return;

        RenderingBlock &b(renderingBlock[block]);

        int xpos = b.upperLeft(0) + x;
        if (xpos > b.lowerRight(0) || xpos >= zRangeX.cols)
            return;

        int ypos = b.upperLeft(1) + y;
        if (ypos > b.lowerRight(1) || ypos >= zRangeX.rows)
            return;

        atomicMin(&zRangeX.ptr(ypos)[xpos], b.zrange(0));
        atomicMax(&zRangeY.ptr(ypos)[xpos], b.zrange(1));

        return;
    }
};

struct RaytracingFunctor
{
    cv::cuda::PtrStepSz<float> zRangeX;
    cv::cuda::PtrStepSz<float> zRangeY;

    int cols, rows;
    float voxelSize;
    float voxelSizeInv;
    float invfx, invfy, cx, cy;
    float raycastStep;
    SE3f T;
    SE3f TInv;

    HashEntry *hashTable;
    Voxel *blocks;
    int numBucket;

    mutable cv::cuda::PtrStep<Vec4f> vmap;

    __device__ __forceinline__ float readSDF(const Vec3f &voxelPos, bool &valid) const
    {
        Voxel *voxel = NULL;
        findVoxel(hashTable, blocks, numBucket, voxelPos.cast<int>(), voxel);
        if (voxel && voxel->wt != 0)
        {
            valid = true;
            return unpackFloat(voxel->sdf);
        }

        valid = false;
        return 0;
    }

    __device__ __forceinline__ float readSDFInterp(const Vec3f &pt, bool &valid) const
    {
        Vec3f xyz;
        xyz(0) = pt(0) - floor(pt(0));
        xyz(1) = pt(1) - floor(pt(1));
        xyz(2) = pt(2) - floor(pt(2));
        float sdf[2], result[4];
        bool validPt;

        sdf[0] = readSDF(pt, validPt);
        sdf[1] = readSDF(pt + Vec3f(1, 0, 0), valid);
        validPt &= valid;
        result[0] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];

        sdf[0] = readSDF(pt + Vec3f(0, 1, 0), valid);
        validPt &= valid;
        sdf[1] = readSDF(pt + Vec3f(1, 1, 0), valid);
        validPt &= valid;

        result[1] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];
        result[2] = (1.0f - xyz(1)) * result[0] + xyz(1) * result[1];

        sdf[0] = readSDF(pt + Vec3f(0, 0, 1), valid);
        validPt &= valid;
        sdf[1] = readSDF(pt + Vec3f(1, 0, 1), valid);
        validPt &= valid;
        result[0] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];

        sdf[0] = readSDF(pt + Vec3f(0, 1, 1), valid);
        validPt &= valid;
        sdf[1] = readSDF(pt + Vec3f(1, 1, 1), valid);
        validPt &= valid;

        result[1] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];
        result[3] = (1.0f - xyz(1)) * result[0] + xyz(1) * result[1];
        valid = validPt;

        return (1.0f - xyz(2)) * result[2] + xyz(2) * result[3];
    }

    __device__ __forceinline__ void operator()() const
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= cols || y >= rows)
            return;

        Vec2s localIdx;
        localIdx(0) = __float2int_rd((float)x / 8);
        localIdx(1) = __float2int_rd((float)y / 8);

        Vec2f zrange;
        zrange(0) = zRangeX.ptr(localIdx(1))[localIdx(0)];
        zrange(1) = zRangeY.ptr(localIdx(1))[localIdx(0)];
        if (zrange(1) < FLT_EPSILON)
            return;

        Vec3f pt = unproject(x, y, zrange(0), invfx, invfy, cx, cy);
        float distStart = pt.norm() * voxelSizeInv;
        Vec3f blockStart = T * pt * voxelSizeInv;

        pt = unproject(x, y, zrange(1), invfx, invfy, cx, cy);
        float distEnd = pt.norm() * voxelSizeInv;
        Vec3f blockEnd = T * pt * voxelSizeInv;

        Vec3f dir = (blockEnd - blockStart).normalized();
        Vec3f result = blockStart;

        bool validSDF = false;
        bool ptFound = false;
        float step;
        float sdf = 1.0f;
        float lastReadSDF;

        while (distStart < distEnd)
        {
            lastReadSDF = sdf;
            sdf = readSDF(result, validSDF);

            if (sdf <= 0.5f && sdf >= -0.5f)
                sdf = readSDFInterp(result, validSDF);
            if (sdf <= 0.0f)
                break;
            if (sdf >= 0.f && lastReadSDF < 0.f)
                return;
            if (validSDF)
                step = max(sdf * raycastStep, 1.0f);
            else
                step = 2;

            result += step * dir;
            distStart += step;
        }

        if (sdf <= 0.0f)
        {
            step = sdf * raycastStep;
            result += step * dir;

            sdf = readSDFInterp(result, validSDF);

            step = sdf * raycastStep;
            result += step * dir;

            if (validSDF)
                ptFound = true;
        }

        if (ptFound)
        {
            result = TInv * result * voxelSize;
            vmap.ptr(y)[x].head<3>() = result;
            vmap.ptr(y)[x](3) = 1.0f;
        }
    }
};