#pragma once
#include "utils/numType.h"
#include "utils/prefixSum.h"
#include "localMapper/denseMap.h"
#include "utils/triangleTable.h"

#define RenderingBlockSizeX 16
#define RenderingBlockSizeY 16
#define RenderingBlockSubSample 8
#define MaxNumRenderingBlock 100000
#define MaxNumTriangle 20000000

__device__ __forceinline__ Vec2f project(
    const Vec3f &pt, const float &fx, const float &fy,
    const float &cx, const float &cy)
{
    Vec2f ptWarped;
    ptWarped(0) = fx * pt(0) / pt(2) + cx;
    ptWarped(1) = fy * pt(1) / pt(2) + cy;
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
    const Vec3f &pt, const SE3f &Tinv,
    const int &cols, const int &rows,
    const float &fx, const float &fy,
    const float &cx, const float &cy,
    const float &depthMin, const float &depthMax)
{
    auto ptTransformed = Tinv * pt;
    auto ptWarped = project(ptTransformed, fx, fy, cx, cy);

    return ptWarped(0) >= 0 && ptWarped(1) >= 0 &&
           ptWarped(0) < cols && ptWarped(1) < rows &&
           ptTransformed(2) >= depthMin &&
           ptTransformed(2) <= depthMax;
}

__device__ __forceinline__ bool checkBlockVisibility(
    const Vec3f &blockPos, const SE3f &Tinv,
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

        if (checkVertexVisibility(tmp * scale, Tinv, cols, rows, fx, fy, cx, cy, depthMin, depthMax))
            return true;
    }

    return false;
}

__device__ __forceinline__ float readSDF(
    const Vec3f &ptWorldScaled, bool &valid,
    HashEntry *hashTable, Voxel *blocks,
    int numBucket)
{
    Voxel *voxel = NULL;
    Vec3i voxelPos;
    voxelPos(0) = floor(ptWorldScaled(0));
    voxelPos(1) = floor(ptWorldScaled(1));
    voxelPos(2) = floor(ptWorldScaled(2));
    findVoxel(hashTable, blocks, numBucket, voxelPos, voxel);

    if (voxel != NULL && voxel->wt != 0)
    {
        valid = true;
        return unpackFloat(voxel->sdf);
    }
    else
    {
        valid = false;
        return 1.0f;
    }
}

struct CreateVoxelBlockFunctor
{
    SE3f T;
    int cols, rows;
    float invfx, invfy;
    float cx, cy;
    float depthMin;
    float depthMax;

    float truncDistHalf;
    float voxelSizeInv;
    int numEntry;
    int numBucket;
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
        if (isnan(dist) || dist < depthMin || dist > depthMax)
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
            Vec3i voxelPos(floor(ptNear(0)), floor(ptNear(1)), floor(ptNear(2)));
            createBlock(
                hashTable,
                bucketMutex,
                heap,
                heapPtr,
                excessPtr,
                numEntry,
                numBucket,
                voxelPosToBlockPos(voxelPos));
            ptNear += dir;
        }
    }
};

struct CheckEntryVisibilityFunctor
{
    SE3f Tinv;
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
                bool visible = checkBlockVisibility(
                    entry.pos.cast<float>(), Tinv,
                    cols, rows,
                    fx, fy,
                    cx, cy,
                    depthMin,
                    depthMax,
                    voxelSize);

                if (visible)
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
    SE3f Tinv;

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
        if (entry.ptr < 0)
            return;

        auto blockPos = entry.pos * BlockSize;

#pragma unroll
        for (int blockIdxZ = 0; blockIdxZ < 8; ++blockIdxZ)
        {
            auto localPos = Vec3i(threadIdx.x, threadIdx.y, blockIdxZ);
            auto pt = Tinv * (blockPos + localPos).cast<float>() * voxelSize;
            int u = __float2int_rd(fx * pt(0) / pt(2) + cx + 0.5f);
            int v = __float2int_rd(fy * pt(1) / pt(2) + cy + 0.5f);
            if (u < 0 || v < 0 || u >= cols || v >= rows)
                continue;

            float dist = depth.ptr(v)[u];
            if (isnan(dist) || dist > depthMax || dist < depthMin)
                continue;

            float newSDF = dist - pt(2);
            if (newSDF < -truncDist)
                continue;

            newSDF = fmin(1.0f, newSDF / truncDist);
            int localIdx = localPosToLocalIdx(localPos);
            Voxel &curr = voxels[entry.ptr + localIdx];

            float oldSDF = unpackFloat(curr.sdf);
            int oldWT = curr.wt;

            if (oldWT == 0)
            {
                curr.sdf = packFloat(newSDF);
                curr.wt = 1;
                continue;
            }
            else
            {
                curr.sdf = packFloat((oldWT * oldSDF + newSDF) / (oldWT + 1));
                curr.wt = min(255, oldWT + 1);
            }
        }
    }
};

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
    SE3f Tinv;
    float fx, fy, cx, cy;

    float scale;
    float depthMin, depthMax;

    uint *numRenderingBlock;
    uint numVisibleEntry;

    HashEntry *visibleEntry;
    RenderingBlock *renderingBlock;
    mutable cv::cuda::PtrStepSz<float> zRangeX;
    mutable cv::cuda::PtrStep<float> zRangeY;

    __device__ __forceinline__ bool projectBlock(const Vec3f &blockPos, RenderingBlock &block) const
    {
        block.upperLeft = Vec2s(zRangeX.cols, zRangeX.rows);
        block.lowerRight = Vec2s(-1, -1);
        block.zrange = Vec2f(depthMax, depthMin);

#pragma unroll
        for (int corner = 0; corner < 8; ++corner)
        {
            Vec3f tmp = blockPos;
            tmp(0) += (corner & 1) ? 1 : 0;
            tmp(1) += (corner & 2) ? 1 : 0;
            tmp(2) += (corner & 4) ? 1 : 0;

            Vec3f ptTransformed = Tinv * tmp * scale;
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
        {
            for (int x = 0; x < nx; ++x)
            {
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
        }
    }

    __device__ __forceinline__ void operator()() const
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        bool valid = false;
        uint requiredNumBlocks = 0;
        RenderingBlock block;
        int nx, ny;

        if (idx < numVisibleEntry && visibleEntry[idx].ptr != -1)
        {
            valid = projectBlock(visibleEntry[idx].pos.cast<float>(), block);
            if (valid)
            {
                float dx = (float)block.lowerRight(0) - block.upperLeft(0) + 1;
                float dy = (float)block.lowerRight(1) - block.upperLeft(1) + 1;
                nx = __float2int_ru(dx / RenderingBlockSizeX);
                ny = __float2int_ru(dy / RenderingBlockSizeY);
                requiredNumBlocks = nx * ny;
                uint totalBlocks = *numRenderingBlock + requiredNumBlocks;
                if (totalBlocks >= MaxNumRenderingBlock)
                    requiredNumBlocks = 0;
            }
        }

        int offset = computeOffset<1024>(requiredNumBlocks, numRenderingBlock);
        if (valid && offset != -1 && (offset + requiredNumBlocks) < MaxNumRenderingBlock)
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
    SE3f Tinv;

    HashEntry *hashTable;
    Voxel *blocks;
    int numBucket;

    mutable cv::cuda::PtrStepSz<Vec4f> vmap;

    __device__ __forceinline__ float readSDFInterp(
        const Vec3f &pt, bool &valid) const
    {
        Vec3f xyz;
        xyz(0) = pt(0) - floor(pt(0));
        xyz(1) = pt(1) - floor(pt(1));
        xyz(2) = pt(2) - floor(pt(2));
        float sdf[2], result[4];
        bool validPt;

        sdf[0] = readSDF(pt, validPt, hashTable, blocks, numBucket);
        sdf[1] = readSDF(pt + Vec3f(1, 0, 0), valid, hashTable, blocks, numBucket);
        validPt &= valid;
        result[0] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];

        sdf[0] = readSDF(pt + Vec3f(0, 1, 0), valid, hashTable, blocks, numBucket);
        validPt &= valid;
        sdf[1] = readSDF(pt + Vec3f(1, 1, 0), valid, hashTable, blocks, numBucket);
        validPt &= valid;

        result[1] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];
        result[2] = (1.0f - xyz(1)) * result[0] + xyz(1) * result[1];

        sdf[0] = readSDF(pt + Vec3f(0, 0, 1), valid, hashTable, blocks, numBucket);
        validPt &= valid;
        sdf[1] = readSDF(pt + Vec3f(1, 0, 1), valid, hashTable, blocks, numBucket);
        validPt &= valid;
        result[0] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];

        sdf[0] = readSDF(pt + Vec3f(0, 1, 1), valid, hashTable, blocks, numBucket);
        validPt &= valid;
        sdf[1] = readSDF(pt + Vec3f(1, 1, 1), valid, hashTable, blocks, numBucket);
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
        if (x >= vmap.cols || y >= vmap.rows)
            return;

        vmap.ptr(y)[x](0) = __int_as_float(0x7fffffff);

        short u = __float2int_rd((float)x / 8);
        short v = __float2int_rd((float)y / 8);

        // float zNear = zRangeX.ptr(v)[u];
        // float zFar = zRangeY.ptr(v)[u];
        float zNear = 0.3f;
        float zFar = 3.0f;
        if (zFar <= zNear)
            return;

        Vec3f pt = unproject(x, y, zNear, invfx, invfy, cx, cy);
        float distStart = pt.norm() * voxelSizeInv;
        Vec3f blockStart = T * pt * voxelSizeInv;

        pt = unproject(x, y, zFar, invfx, invfy, cx, cy);
        float distEnd = pt.norm() * voxelSizeInv;
        Vec3f blockEnd = T * pt * voxelSizeInv;

        Vec3f dir = (blockEnd - blockStart).normalized();
        Vec3f result = blockStart;

        bool validSDF = false;
        bool ptFound = false;
        float step;
        float sdf = 1.0f;
        float lastReadSDF = 1.0f;

        while (distStart < distEnd)
        {
            lastReadSDF = sdf;
            sdf = readSDF(result, validSDF, hashTable, blocks, numBucket);

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
            lastReadSDF = sdf;
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
            Vec3f worldPt = Tinv * result * voxelSize;
            vmap.ptr(y)[x].head<3>() = worldPt;
            vmap.ptr(y)[x](3) = 1.0f;
        }
        else
            vmap.ptr(y)[x](3) = -1.0f;
    }
};

struct GenerateMeshFunctor
{
    Vec3f *triangles;
    uint *numTriangle;
    Vec3f *surfaceNormal;

    uint numVisibleBlock;
    HashEntry *visibleEntry;
    HashEntry *hashTable;
    Voxel *blocks;
    int numBucket;
    float voxelSize;

    __device__ __forceinline__ bool readAdjecentSDF(
        float *sdf, const Vec3f &ptWorldScaled) const
    {
        bool valid = false;
        sdf[0] = readSDF(ptWorldScaled, valid, hashTable, blocks, numBucket);
        if (!valid)
            return false;
        sdf[1] = readSDF(ptWorldScaled + Vec3f(1, 0, 0), valid, hashTable, blocks, numBucket);
        if (!valid)
            return false;
        sdf[2] = readSDF(ptWorldScaled + Vec3f(1, 1, 0), valid, hashTable, blocks, numBucket);
        if (!valid)
            return false;
        sdf[3] = readSDF(ptWorldScaled + Vec3f(0, 1, 0), valid, hashTable, blocks, numBucket);
        if (!valid)
            return false;
        sdf[4] = readSDF(ptWorldScaled + Vec3f(0, 0, 1), valid, hashTable, blocks, numBucket);
        if (!valid)
            return false;
        sdf[5] = readSDF(ptWorldScaled + Vec3f(1, 0, 1), valid, hashTable, blocks, numBucket);
        if (!valid)
            return false;
        sdf[6] = readSDF(ptWorldScaled + Vec3f(1, 1, 1), valid, hashTable, blocks, numBucket);
        if (!valid)
            return false;
        sdf[7] = readSDF(ptWorldScaled + Vec3f(0, 1, 1), valid, hashTable, blocks, numBucket);
        if (!valid)
            return false;
        return true;
    }

    __device__ __forceinline__ float interpolateLinear(float &v1, float &v2) const
    {
        if (fabs(0 - v1) < 1e-6)
            return 0;
        if (fabs(0 - v2) < 1e-6)
            return 1;
        if (fabs(v1 - v2) < 1e-6)
            return 0;
        return (0 - v1) / (v2 - v1);
    }

    __device__ __forceinline__ int getVertexArray(Vec3f *array, const Vec3f &voxelPos) const
    {
        float sdf[8];

        if (!readAdjecentSDF(sdf, voxelPos))
            return -1;

        int cubeIdx = 0;
        if (sdf[0] < 0)
            cubeIdx |= 1;
        if (sdf[1] < 0)
            cubeIdx |= 2;
        if (sdf[2] < 0)
            cubeIdx |= 4;
        if (sdf[3] < 0)
            cubeIdx |= 8;
        if (sdf[4] < 0)
            cubeIdx |= 16;
        if (sdf[5] < 0)
            cubeIdx |= 32;
        if (sdf[6] < 0)
            cubeIdx |= 64;
        if (sdf[7] < 0)
            cubeIdx |= 128;

        if (edgeTable[cubeIdx] == 0)
            return -1;

        if (edgeTable[cubeIdx] & 1)
        {
            float val = interpolateLinear(sdf[0], sdf[1]);
            array[0] = voxelPos + Vec3f(val, 0, 0);
        }
        if (edgeTable[cubeIdx] & 2)
        {
            float val = interpolateLinear(sdf[1], sdf[2]);
            array[1] = voxelPos + Vec3f(1, val, 0);
        }
        if (edgeTable[cubeIdx] & 4)
        {
            float val = interpolateLinear(sdf[2], sdf[3]);
            array[2] = voxelPos + Vec3f(1 - val, 1, 0);
        }
        if (edgeTable[cubeIdx] & 8)
        {
            float val = interpolateLinear(sdf[3], sdf[0]);
            array[3] = voxelPos + Vec3f(0, 1 - val, 0);
        }
        if (edgeTable[cubeIdx] & 16)
        {
            float val = interpolateLinear(sdf[4], sdf[5]);
            array[4] = voxelPos + Vec3f(val, 0, 1);
        }
        if (edgeTable[cubeIdx] & 32)
        {
            float val = interpolateLinear(sdf[5], sdf[6]);
            array[5] = voxelPos + Vec3f(1, val, 1);
        }
        if (edgeTable[cubeIdx] & 64)
        {
            float val = interpolateLinear(sdf[6], sdf[7]);
            array[6] = voxelPos + Vec3f(1 - val, 1, 1);
        }
        if (edgeTable[cubeIdx] & 128)
        {
            float val = interpolateLinear(sdf[7], sdf[4]);
            array[7] = voxelPos + Vec3f(0, 1 - val, 1);
        }
        if (edgeTable[cubeIdx] & 256)
        {
            float val = interpolateLinear(sdf[0], sdf[4]);
            array[8] = voxelPos + Vec3f(0, 0, val);
        }
        if (edgeTable[cubeIdx] & 512)
        {
            float val = interpolateLinear(sdf[1], sdf[5]);
            array[9] = voxelPos + Vec3f(1, 0, val);
        }
        if (edgeTable[cubeIdx] & 1024)
        {
            float val = interpolateLinear(sdf[2], sdf[6]);
            array[10] = voxelPos + Vec3f(1, 1, val);
        }
        if (edgeTable[cubeIdx] & 2048)
        {
            float val = interpolateLinear(sdf[3], sdf[7]);
            array[11] = voxelPos + Vec3f(0, 1, val);
        }

        return cubeIdx;
    }

    __device__ __forceinline__ void operator()() const
    {
        int idx = blockIdx.y * gridDim.x + blockIdx.x;
        if (*numTriangle >= MaxNumTriangle || idx >= numVisibleBlock)
            return;

        Vec3f array[12];
        Vec3i voxelPos = visibleEntry[idx].pos * BlockSize;

        for (int voxelIdZ = 0; voxelIdZ < BlockSize; ++voxelIdZ)
        {
            Vec3i localPos = Vec3i(threadIdx.x, threadIdx.y, voxelIdZ);
            int cubeIdx = getVertexArray(array, (voxelPos + localPos).cast<float>());
            if (cubeIdx <= 0)
                continue;

            for (int i = 0; triTable[cubeIdx][i] != -1; i += 3)
            {
                uint triangleId = atomicAdd(numTriangle, 1);
                if (triangleId >= MaxNumTriangle)
                    return;

                triangles[triangleId * 3] = array[triTable[cubeIdx][i]] * voxelSize;
                triangles[triangleId * 3 + 1] = array[triTable[cubeIdx][i + 1]] * voxelSize;
                triangles[triangleId * 3 + 2] = array[triTable[cubeIdx][i + 2]] * voxelSize;

                auto v10 = triangles[triangleId * 3 + 1] - triangles[triangleId * 3];
                auto v20 = triangles[triangleId * 3 + 2] - triangles[triangleId * 3];
                auto n = v10.cross(v20).normalized();
                surfaceNormal[triangleId * 3] = n;
                surfaceNormal[triangleId * 3 + 1] = n;
                surfaceNormal[triangleId * 3 + 2] = n;
            }
        }
    }
};