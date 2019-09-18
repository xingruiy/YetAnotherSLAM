#include "map_proc.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <thrust/device_vector.h>
#include "utils/numType.h"
#include "utils/cudaUtils.h"

__device__ inline bool is_vertex_visible(
    Vec3f pt, SE3f inv_pose,
    int cols, int rows, float fx,
    float fy, float cx, float cy,
    float depthMin, float depthMax)
{
    pt = inv_pose * (pt);
    Vec2f pt2d = Vec2f(fx * pt(0) / pt(2) + cx, fy * pt(1) / pt(2) + cy);
    return !(pt2d(0) < 0 || pt2d(1) < 0 ||
             pt2d(0) > cols - 1 || pt2d(1) > rows - 1 ||
             pt(2) < depthMin || pt(2) > depthMax);
}

__device__ inline bool is_block_visible(
    const Vec3i &block_pos,
    SE3f inv_pose, const float &voxelSize,
    int cols, int rows, float fx,
    float fy, float cx, float cy,
    float depthMin, float depthMax)
{
    float scale = voxelSize * BLOCK_SIZE; //param.block_size_metric();
#pragma unroll
    for (int corner = 0; corner < 8; ++corner)
    {
        Vec3i tmp = block_pos;
        tmp(0) += (corner & 1) ? 1 : 0;
        tmp(1) += (corner & 2) ? 1 : 0;
        tmp(2) += (corner & 4) ? 1 : 0;

        if (is_vertex_visible(tmp.cast<float>() * scale, inv_pose, cols, rows, fx, fy, cx, cy, depthMin, depthMax))
            return true;
    }

    return false;
}

__global__ void check_visibility_flag_kernel(
    MapStruct map_struct, uchar *flag, SE3f inv_pose,
    int cols, int rows, float fx, float fy, float cx, float cy, float voxelSize,
    float depthMin, float depthMax)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= map_struct.hashTableSize)
        return;

    HashEntry &current = map_struct.hash_table_[idx];
    if (current.ptr_ != -1)
    {
        switch (flag[idx])
        {
        default:
        {
            if (is_block_visible(current.pos_, inv_pose, voxelSize, cols, rows, fx, fy, cx, cy, depthMin, depthMax))
            {
                flag[idx] = 1;
            }
            else
            {
                current.ptr_ = -1;
                flag[idx] = 0;
            }

            return;
        }
        case 2:
        {
            flag[idx] = 1;
            return;
        }
        }
    }
}

__global__ void copy_visible_block_kernel(HashEntry *hash_table, HashEntry *visible_block, int hashTableSize, const uchar *flag, const int *pos)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= hashTableSize)
        return;

    if (flag[idx] == 1)
        visible_block[pos[idx]] = hash_table[idx];
}

__device__ inline Vec2f project(
    Vec3f pt, float fx, float fy, float cx, float cy)
{
    return Vec2f(fx * pt(0) / pt(2) + cx, fy * pt(1) / pt(2) + cy);
}

__device__ inline Vec3f unproject(
    int x, int y, float z, float invfx, float invfy, float cx, float cy)
{
    return Vec3f(invfx * (x - cx) * z, invfy * (y - cy) * z, z);
}

__device__ inline Vec3f unproject_world(
    int x, int y, float z, float invfx,
    float invfy, float cx, float cy, SE3f pose)
{
    return pose * (unproject(x, y, z, invfx, invfy, cx, cy));
}

__global__ void create_blocks_kernel(MapStruct map_struct, cv::cuda::PtrStepSz<float> depth,
                                     float invfx, float invfy, float cx, float cy,
                                     SE3f pose, uchar *flag, float depthMin, float depthMax)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= depth.cols || y >= depth.rows)
        return;

    float z = depth.ptr(y)[x];
    if (isnan(z) || z < depthMin || z > depthMax)
        return;

    float z_thresh = map_struct.truncationDist * 0.5;
    float z_near = max(depthMin, z - z_thresh);
    float z_far = min(depthMax, z + z_thresh);
    if (z_near >= z_far)
        return;

    Vec3i block_near = voxelPosToBlockPos(worldPtToVoxelPos(unproject_world(x, y, z_near, invfx, invfy, cx, cy, pose), map_struct.voxelSize));
    Vec3i block_far = voxelPosToBlockPos(worldPtToVoxelPos(unproject_world(x, y, z_far, invfx, invfy, cx, cy, pose), map_struct.voxelSize));

    Vec3i d = block_far - block_near;
    Vec3i increment = Vec3i(d(0) < 0 ? -1 : 1, d(1) < 0 ? -1 : 1, d(2) < 0 ? -1 : 1);
    Vec3i incre_abs = Vec3i(abs(d(0)), abs(d(1)), abs(d(2)));
    Vec3i incre_err = Vec3i(incre_abs(0) << 1, incre_abs(1) << 1, incre_abs(2) << 1);

    int err_1;
    int err_2;

    // Bresenham's line algorithm
    // details see : https://en.m.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    if ((incre_abs(0) >= incre_abs(1)) && (incre_abs(0) >= incre_abs(2)))
    {
        err_1 = incre_err(1) - 1;
        err_2 = incre_err(2) - 1;
        createBlock(block_near,
                    map_struct.heap_mem_,
                    map_struct.heap_mem_counter_,
                    map_struct.hash_table_,
                    map_struct.bucket_mutex_,
                    map_struct.excess_counter_,
                    map_struct.hashTableSize,
                    map_struct.bucketSize);
        for (int i = 0; i < incre_abs(0); ++i)
        {
            if (err_1 > 0)
            {
                block_near(1) += increment(1);
                err_1 -= incre_err(0);
            }

            if (err_2 > 0)
            {
                block_near(2) += increment(2);
                err_2 -= incre_err(0);
            }

            err_1 += incre_err(1);
            err_2 += incre_err(2);
            block_near(0) += increment(0);
            createBlock(block_near,
                        map_struct.heap_mem_,
                        map_struct.heap_mem_counter_,
                        map_struct.hash_table_,
                        map_struct.bucket_mutex_,
                        map_struct.excess_counter_,
                        map_struct.hashTableSize,
                        map_struct.bucketSize);
        }
    }
    else if ((incre_abs(1) >= incre_abs(0)) && (incre_abs(1) >= incre_abs(2)))
    {
        err_1 = incre_err(0) - 1;
        err_2 = incre_err(2) - 1;
        createBlock(block_near,
                    map_struct.heap_mem_,
                    map_struct.heap_mem_counter_,
                    map_struct.hash_table_,
                    map_struct.bucket_mutex_,
                    map_struct.excess_counter_,
                    map_struct.hashTableSize,
                    map_struct.bucketSize);
        for (int i = 0; i < incre_abs(1); ++i)
        {
            if (err_1 > 0)
            {
                block_near(0) += increment(0);
                err_1 -= incre_err(1);
            }

            if (err_2 > 0)
            {
                block_near(2) += increment(2);
                err_2 -= incre_err(1);
            }

            err_1 += incre_err(0);
            err_2 += incre_err(2);
            block_near(1) += increment(1);
            createBlock(block_near,
                        map_struct.heap_mem_,
                        map_struct.heap_mem_counter_,
                        map_struct.hash_table_,
                        map_struct.bucket_mutex_,
                        map_struct.excess_counter_,
                        map_struct.hashTableSize,
                        map_struct.bucketSize);
        }
    }
    else
    {
        err_1 = incre_err(1) - 1;
        err_2 = incre_err(0) - 1;
        createBlock(block_near,
                    map_struct.heap_mem_,
                    map_struct.heap_mem_counter_,
                    map_struct.hash_table_,
                    map_struct.bucket_mutex_,
                    map_struct.excess_counter_,
                    map_struct.hashTableSize,
                    map_struct.bucketSize);
        for (int i = 0; i < incre_abs(2); ++i)
        {
            if (err_1 > 0)
            {
                block_near(1) += increment(1);
                err_1 -= incre_err(2);
            }

            if (err_2 > 0)
            {
                block_near(0) += increment(0);
                err_2 -= incre_err(2);
            }

            err_1 += incre_err(1);
            err_2 += incre_err(0);
            block_near(2) += increment(2);
            createBlock(block_near,
                        map_struct.heap_mem_,
                        map_struct.heap_mem_counter_,
                        map_struct.hash_table_,
                        map_struct.bucket_mutex_,
                        map_struct.excess_counter_,
                        map_struct.hashTableSize,
                        map_struct.bucketSize);
        }
    }
}

__global__ void update_map_kernel(MapStruct map_struct,
                                  HashEntry *visible_blocks,
                                  uint count_visible_block,
                                  cv::cuda::PtrStepSz<float> depth,
                                  SE3f inv_pose,
                                  float fx, float fy,
                                  float cx, float cy,
                                  float depthMin, float depthMax)
{
    if (blockIdx.x >= map_struct.hashTableSize || blockIdx.x >= count_visible_block)
        return;

    HashEntry &current = visible_blocks[blockIdx.x];

    Vec3i voxel_pos = blockPosToVoxelPos(current.pos_);
    float dist_thresh = map_struct.truncationDist;
    float inv_dist_thresh = 1.0 / dist_thresh;

#pragma unroll
    for (int block_idx_z = 0; block_idx_z < 8; ++block_idx_z)
    {
        Vec3i local_pos = Vec3i(threadIdx.x, threadIdx.y, block_idx_z);
        Vec3f pt = inv_pose * (voxelPosToWorldPt(voxel_pos + local_pos, map_struct.voxelSize));

        int u = __float2int_rd(fx * pt(0) / pt(2) + cx + 0.5);
        int v = __float2int_rd(fy * pt(1) / pt(2) + cy + 0.5);
        if (u < 0 || v < 0 || u > depth.cols - 1 || v > depth.rows - 1)
            continue;

        float dist = depth.ptr(v)[u];
        if (isnan(dist) || dist < 1e-2 || dist > depthMax || dist < depthMin)
            continue;

        float sdf = dist - pt(2);
        if (sdf < -dist_thresh)
            continue;

        sdf = fmin(1.0f, sdf * inv_dist_thresh);
        const int local_idx = localPosToLocalIdx(local_pos);
        Voxel &voxel = map_struct.voxels_[current.ptr_ + local_idx];

        auto sdf_p = unpackFloat(voxel.sdf);
        auto weight_p = voxel.weight;
        auto weight = 1 / (dist);

        if (weight_p == 0)
        {
            voxel.sdf = packFloat(sdf);
            voxel.weight = weight;
            continue;
        }

        // fuse depth
        sdf_p = (sdf_p * weight_p + sdf * weight) / (weight_p + weight);
        voxel.sdf = packFloat(sdf_p);
        voxel.weight = (weight_p + weight);
    }
}

void update(
    MapStruct map_struct,
    const GMat depth,
    const SE3 &frame_pose,
    const Mat33d &K,
    GMat &cv_flag,
    GMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count)
{
    if (cv_flag.empty())
        cv_flag.create(1, map_struct.hashTableSize, CV_8UC1);
    if (cv_pos_array.empty())
        cv_pos_array.create(1, map_struct.hashTableSize, CV_32SC1);

    thrust::device_ptr<uchar> flag(cv_flag.ptr<uchar>());
    thrust::device_ptr<int> pos_array(cv_pos_array.ptr<int>());

    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float invfx = 1.0 / fx;
    float invfy = 1.0 / fy;

    const int cols = depth.cols;
    const int rows = depth.rows;

    dim3 thread(8, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    create_blocks_kernel<<<block, thread>>>(
        map_struct,
        depth,
        invfx,
        invfy,
        cx, cy,
        frame_pose.cast<float>(),
        flag.get(),
        0.1f,
        3.0f);

    thread = dim3(1024);
    block = dim3(div_up(map_struct.hashTableSize, thread.x));

    check_visibility_flag_kernel<<<block, thread>>>(
        map_struct,
        flag.get(),
        frame_pose.inverse().cast<float>(),
        cols, rows,
        fx, fy,
        cx, cy,
        map_struct.voxelSize,
        0.1f,
        3.0f);

    thrust::exclusive_scan(flag, flag + map_struct.hashTableSize, pos_array);

    copy_visible_block_kernel<<<block, thread>>>(
        map_struct.hash_table_,
        visible_blocks,
        map_struct.hashTableSize,
        flag.get(),
        pos_array.get());

    visible_block_count = pos_array[map_struct.hashTableSize - 1];

    if (visible_block_count == 0)
        return;

    thread = dim3(8, 8);
    block = dim3(visible_block_count);

    update_map_kernel<<<block, thread>>>(
        map_struct,
        visible_blocks,
        visible_block_count,
        depth,
        frame_pose.inverse().cast<float>(),
        fx, fy,
        cx, cy,
        0.1f,
        3.0f);
}
