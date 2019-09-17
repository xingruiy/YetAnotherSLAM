#include "utils/numType.h"
#include "utils/prefixSum.h"
#include "localMapper/denseMap.h"

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

struct RenderingBlockDelegate
{
    SE3f TInv;
    int cols, rows;
    float fx, fy, cx, cy;

    float scale;
    float depthMin, depthMax;

    uint *numRenderingBlock;
    uint numVisibleEntry;
    uint numMaxRenderingBlock;

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
                if (offset < numMaxRenderingBlock)
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
                if (totalBlocks >= numMaxRenderingBlock)
                    requiredBlocks = 0;
            }
        }

        int offset = computeOffset<1024>(requiredBlocks, numRenderingBlock);
        if (valid && offset != -1 && (offset + requiredBlocks) < numMaxRenderingBlock)
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

// __global__ void __launch_bounds__(32, 16) raycast_kernel(MapRenderingDelegate delegate)
// {
//     delegate();
// }

// __global__ void __launch_bounds__(32, 16) raycast_with_colour_kernel(MapRenderingDelegate delegate)
// {
//     delegate.raycast_with_colour();
// }

// void raycast(MapStorage map_struct,
//              MapState state,
//              cv::cuda::GpuMat vmap,
//              cv::cuda::GpuMat nmap,
//              cv::cuda::GpuMat zRangeX,
//              cv::cuda::GpuMat zRangeY,
//              const Sophus::SE3d &pose,
//              const IntrinsicMatrix intrinsic_matrix)
// {
//     const int cols = vmap.cols;
//     const int rows = vmap.rows;

//     MapRenderingDelegate delegate;

//     delegate.cols = cols;
//     delegate.rows = rows;
//     delegate.map_struct = map_struct;
//     delegate.vmap = vmap;
//     delegate.nmap = nmap;
//     delegate.zRangeX = zRangeX;
//     delegate.zRangeY = zRangeY;
//     delegate.invfx = intrinsic_matrix.invfx;
//     delegate.invfy = intrinsic_matrix.invfy;
//     delegate.cx = intrinsic_matrix.cx;
//     delegate.cy = intrinsic_matrix.cy;
//     delegate.pose = pose.cast<float>().matrix3x4();
//     delegate.inv_pose = pose.inverse().cast<float>().matrix3x4();

//     dim3 thread(4, 8);
//     dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

//     call_device_functor<<<block, thread>>>(delegate);
// }

// void raycast_with_colour(MapStorage map_struct,
//                          MapState state,
//                          cv::cuda::GpuMat vmap,
//                          cv::cuda::GpuMat nmap,
//                          cv::cuda::GpuMat image,
//                          cv::cuda::GpuMat zRangeX,
//                          cv::cuda::GpuMat zRangeY,
//                          const Sophus::SE3d &pose,
//                          const IntrinsicMatrix intrinsic_matrix)
// {
//     const int cols = vmap.cols;
//     const int rows = vmap.rows;

//     MapRenderingDelegate delegate;

//     delegate.cols = cols;
//     delegate.rows = rows;
//     delegate.map_struct = map_struct;
//     delegate.vmap = vmap;
//     delegate.nmap = nmap;
//     delegate.image = image;
//     delegate.zRangeX = zRangeX;
//     delegate.zRangeY = zRangeY;
//     delegate.invfx = intrinsic_matrix.invfx;
//     delegate.invfy = intrinsic_matrix.invfy;
//     delegate.cx = intrinsic_matrix.cx;
//     delegate.cy = intrinsic_matrix.cy;
//     delegate.pose = pose.cast<float>().matrix3x4();
//     delegate.inv_pose = pose.inverse().cast<float>().matrix3x4();

//     dim3 thread(4, 8);
//     dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

//     call_device_functor<<<block, thread>>>(delegate);
// }

// __device__ __forceinline__ bool is_vertex_visible(
//     Vec3f pt, Matrix3x4f inv_pose,
//     int cols, int rows, float fx,
//     float fy, float cx, float cy)
// {
//     pt = inv_pose(pt);
//     Vector2f pt2d = Vector2f(fx * pt.x / pt.z + cx, fy * pt.y / pt.z + cy);
//     return !(ptWarped(0) < 0 || ptWarped(1) < 0 ||
//              ptWarped(0) > cols - 1 || ptWarped(1) > rows - 1 ||
//              pt.z < param.zmin_update || pt.z > param.zmax_update);
// }

// __device__ __forceinline__ bool is_block_visible(
//     const Vector3i &block_pos,
//     const Matrix3x4f &inv_pose,
//     int cols, int rows, float fx,
//     float fy, float cx, float cy)
// {
//     float scale = param.block_size_metric();
// #pragma unroll
//     for (int corner = 0; corner < 8; ++corner)
//     {
//         Vector3i tmp = block_pos;
//         tmp.x += (corner & 1) ? 1 : 0;
//         tmp.y += (corner & 2) ? 1 : 0;
//         tmp.z += (corner & 4) ? 1 : 0;

//         if (is_vertex_visible(tmp * scale, inv_pose, cols, rows, fx, fy, cx, cy))
//             return true;
//     }

//     return false;
// }