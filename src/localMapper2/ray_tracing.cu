#include "localMapper2/denseMap.h"
#include "utils/cudaUtils.h"
#include "utils/prefixSum.h"
#include <opencv2/opencv.hpp>

#define RENDERING_BLOCK_SIZE_X 16
#define RENDERING_BLOCK_SIZE_Y 16
#define RENDERING_BLOCK_SUBSAMPLE 8

struct RenderingBlockDelegate
{
    int width, height;
    SE3f inv_pose;
    float fx, fy, cx, cy;

    uint *rendering_block_count;
    uint visible_block_count;

    HashEntry *visible_block_pos;
    mutable cv::cuda::PtrStepSz<float> zrange_x;
    mutable cv::cuda::PtrStep<float> zrange_y;
    RenderingBlock *rendering_blocks;

    __device__ __forceinline__ Vec2f project(const Vec3f &pt) const
    {
        return Vec2f(fx * pt(0) / pt(2) + cx, fy * pt(1) / pt(2) + cy);
    }

    // compare val with the old value stored in *add
    // and write the bigger one to *add
    __device__ __forceinline__ void atomic_max(float *add, float val) const
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
    __device__ __forceinline__ void atomic_min(float *add, float val) const
    {
        int *address_as_i = (int *)add;
        int old = *address_as_i, assumed;
        do
        {
            assumed = old;
            old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }

    __device__ __forceinline__ bool create_rendering_block(const Vec3i &block_pos, RenderingBlock &block) const
    {
        block.upper_left = Vec2s(zrange_x.cols, zrange_x.rows);
        block.lower_right = Vec2s(-1, -1);
        block.zrange = Vec2f(param.zmax_raycast, param.zmin_raycast);

#pragma unroll
        for (int corner = 0; corner < 8; ++corner)
        {
            Vec3i tmp = block_pos;
            tmp(0) += (corner & 1) ? 1 : 0;
            tmp(1) += (corner & 2) ? 1 : 0;
            tmp(2) += (corner & 4) ? 1 : 0;

            Vec3f pt3d = tmp.cast<float>() * param.block_size_metric();
            pt3d = inv_pose * (pt3d);

            Vec2f pt2d = project(pt3d) / RENDERING_BLOCK_SUBSAMPLE;

            if (block.upper_left(0) > std::floor(pt2d(0)))
                block.upper_left(0) = (int)std::floor(pt2d(0));

            if (block.lower_right(0) < ceil(pt2d(0)))
                block.lower_right(0) = (int)ceil(pt2d(0));

            if (block.upper_left(1) > std::floor(pt2d(1)))
                block.upper_left(1) = (int)std::floor(pt2d(1));

            if (block.lower_right(1) < ceil(pt2d(1)))
                block.lower_right(1) = (int)ceil(pt2d(1));

            if (block.zrange(0) > pt3d(2))
                block.zrange(0) = pt3d(2);

            if (block.zrange(1) < pt3d(2))
                block.zrange(1) = pt3d(2);
        }

        if (block.upper_left(0) < 0)
            block.upper_left(0) = 0;

        if (block.upper_left(1) < 0)
            block.upper_left(1) = 0;

        if (block.lower_right(0) >= zrange_x.cols)
            block.lower_right(0) = zrange_x.cols - 1;

        if (block.lower_right(1) >= zrange_x.rows)
            block.lower_right(1) = zrange_x.rows - 1;

        if (block.upper_left(0) > block.lower_right(0))
            return false;

        if (block.upper_left(1) > block.lower_right(1))
            return false;

        if (block.zrange(0) < param.zmin_raycast)
            block.zrange(0) = param.zmin_raycast;

        if (block.zrange(1) < param.zmin_raycast)
            return false;

        return true;
    }

    __device__ __forceinline__ void create_rendering_block_list(int offset, const RenderingBlock &block, int &nx, int &ny) const
    {
        for (int y = 0; y < ny; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                if (offset < param.num_max_rendering_blocks_)
                {
                    RenderingBlock &b(rendering_blocks[offset++]);
                    b.upper_left(0) = block.upper_left(0) + x * RENDERING_BLOCK_SIZE_X;
                    b.upper_left(1) = block.upper_left(1) + y * RENDERING_BLOCK_SIZE_Y;
                    b.lower_right(0) = block.upper_left(0) + (x + 1) * RENDERING_BLOCK_SIZE_X;
                    b.lower_right(1) = block.upper_left(1) + (y + 1) * RENDERING_BLOCK_SIZE_Y;

                    if (b.lower_right(0) > block.lower_right(0))
                        b.lower_right(0) = block.lower_right(0);

                    if (b.lower_right(1) > block.lower_right(1))
                        b.lower_right(1) = block.lower_right(1);

                    b.zrange = block.zrange;
                }
            }
        }
    }

    __device__ __forceinline__ void operator()() const
    {
        int x = threadIdx.x + blockDim.x * blockIdx.x;

        bool valid = false;
        uint requiredNoBlocks = 0;
        RenderingBlock block;
        int nx, ny;

        if (x < visible_block_count && visible_block_pos[x].ptr_ != -1)
        {
            valid = create_rendering_block(visible_block_pos[x].pos_, block);
            float dx = (float)block.lower_right(0) - block.upper_left(0) + 1;
            float dy = (float)block.lower_right(1) - block.upper_left(1) + 1;
            nx = __float2int_ru(dx / RENDERING_BLOCK_SIZE_X);
            ny = __float2int_ru(dy / RENDERING_BLOCK_SIZE_Y);

            if (valid)
            {
                requiredNoBlocks = nx * ny;
                uint totalNoBlocks = *rendering_block_count + requiredNoBlocks;
                if (totalNoBlocks >= param.num_max_rendering_blocks_)
                {
                    requiredNoBlocks = 0;
                }
            }
        }

        int offset = computeOffset<1024>(requiredNoBlocks, rendering_block_count);
        if (valid && offset != -1 && (offset + requiredNoBlocks) < param.num_max_rendering_blocks_)
            create_rendering_block_list(offset, block, nx, ny);
    }

    __device__ __forceinline__ void fill_rendering_blocks() const
    {
        int x = threadIdx.x;
        int y = threadIdx.y;

        int block = blockIdx.x * 4 + blockIdx.y;
        if (block >= param.num_max_rendering_blocks_)
            return;

        RenderingBlock &b(rendering_blocks[block]);

        int xpos = b.upper_left(0) + x;
        if (xpos > b.lower_right(0) || xpos >= zrange_x.cols)
            return;

        int ypos = b.upper_left(1) + y;
        if (ypos > b.lower_right(1) || ypos >= zrange_x.rows)
            return;

        atomic_min(&zrange_x.ptr(ypos)[xpos], b.zrange(0));
        atomic_max(&zrange_y.ptr(ypos)[xpos], b.zrange(1));

        return;
    }
};

__global__ void create_rendering_blocks_kernel(const RenderingBlockDelegate delegate)
{
    delegate();
}

__global__ void split_and_fill_rendering_blocks_kernel(const RenderingBlockDelegate delegate)
{
    delegate.fill_rendering_blocks();
}

void create_rendering_blocks(
    uint count_visible_block,
    uint &count_rendering_block,
    HashEntry *visible_blocks,
    cv::cuda::GpuMat &zrange_x,
    cv::cuda::GpuMat &zrange_y,
    RenderingBlock *rendering_blocks,
    const Sophus::SE3d &frame_pose,
    const Eigen::Matrix3d &cam_params)
{
    if (count_visible_block == 0)
        return;

    const int cols = zrange_x.cols;
    const int rows = zrange_y.rows;

    zrange_x.setTo(cv::Scalar(100.f));
    zrange_y.setTo(cv::Scalar(0));

    uint *count_device;
    count_rendering_block = 0;
    cudaMalloc((void **)&count_device, sizeof(uint));
    cudaMemset((void *)count_device, 0, sizeof(uint));

    RenderingBlockDelegate delegate;

    delegate.width = cols;
    delegate.height = rows;
    delegate.inv_pose = frame_pose.inverse().cast<float>();
    delegate.zrange_x = zrange_x;
    delegate.zrange_y = zrange_y;
    delegate.fx = cam_params(0, 0);
    delegate.fy = cam_params(1, 1);
    delegate.cx = cam_params(0, 2);
    delegate.cy = cam_params(1, 2);
    delegate.visible_block_pos = visible_blocks;
    delegate.visible_block_count = count_visible_block;
    delegate.rendering_block_count = count_device;
    delegate.rendering_blocks = rendering_blocks;

    dim3 thread = dim3(1024);
    dim3 block = dim3(div_up(count_visible_block, thread.x));

    callDeviceFunctor<<<block, thread>>>(delegate);

    (cudaMemcpy(&count_rendering_block, count_device, sizeof(uint), cudaMemcpyDeviceToHost));
    if (count_rendering_block == 0)
        return;

    thread = dim3(RENDERING_BLOCK_SIZE_X, RENDERING_BLOCK_SIZE_Y);
    block = dim3((uint)ceil((float)count_rendering_block / 4), 4);

    split_and_fill_rendering_blocks_kernel<<<block, thread>>>(delegate);
    (cudaFree((void *)count_device));
}

struct MapRenderingDelegate
{
    int width, height;
    MapStorage map_struct;
    mutable cv::cuda::PtrStep<Vec4f> vmap;
    mutable cv::cuda::PtrStep<Vec4f> nmap;
    mutable cv::cuda::PtrStep<Vec3b> image;
    cv::cuda::PtrStepSz<float> zrange_x;
    cv::cuda::PtrStepSz<float> zrange_y;
    float invfx, invfy, cx, cy;
    SE3f pose, inv_pose;

    __device__ __forceinline__ float read_sdf(const Vec3f &pt3d, bool &valid) const
    {
        Voxel *voxel = NULL;
        findVoxel(map_struct, floor(pt3d), voxel);
        if (voxel && voxel->weight != 0)
        {
            valid = true;
            return voxel->getSDF();
        }
        else
        {
            valid = false;
            return nanf("0x7fffff");
        }
    }

    __device__ __forceinline__ float read_sdf_interped(const Vec3f &pt, bool &valid) const
    {
        // Vec3f xyz = pt - floor(pt);
        Vec3f xyz = Vec3f(pt(0) - floor(pt(0)), pt(1) - floor(pt(1)), pt(2) - floor(pt(2)));
        float sdf[2], result[4];
        bool valid_pt;

        sdf[0] = read_sdf(pt, valid_pt);
        sdf[1] = read_sdf(pt + Vec3f(1, 0, 0), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];

        sdf[0] = read_sdf(pt + Vec3f(0, 1, 0), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Vec3f(1, 1, 0), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];
        result[2] = (1.0f - xyz(1)) * result[0] + xyz(1) * result[1];

        sdf[0] = read_sdf(pt + Vec3f(0, 0, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Vec3f(1, 0, 1), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];

        sdf[0] = read_sdf(pt + Vec3f(0, 1, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Vec3f(1, 1, 1), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];
        result[3] = (1.0f - xyz(1)) * result[0] + xyz(1) * result[1];
        valid = valid_pt;
        return (1.0f - xyz(2)) * result[2] + xyz(2) * result[3];
    }

    __device__ __forceinline__ Vec3f unproject(const int &x, const int &y, const float &z) const
    {
        return Vec3f((x - cx) * invfx * z, (y - cy) * invfy * z, z);
    }

    __device__ __forceinline__ void operator()() const
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= width || y >= height)
            return;

        vmap.ptr(y)[x](0) = __int_as_float(0x7fffffff);

        Vec2i local_id;
        local_id(0) = __float2int_rd((float)x / 8);
        local_id(1) = __float2int_rd((float)y / 8);

        Vec2f zrange;
        zrange(0) = zrange_x.ptr(local_id(1))[local_id(0)];
        zrange(1) = zrange_y.ptr(local_id(1))[local_id(0)];
        if (zrange(1) < 1e-3 || zrange(0) < 1e-3 || isnan(zrange(0)) || isnan(zrange(1)))
            return;

        float sdf = 1.0f;
        float last_sdf;

        Vec3f pt = unproject(x, y, zrange(0));
        float dist_s = pt.norm() * param.inverse_voxel_size();
        Vec3f block_s = pose * (pt)*param.inverse_voxel_size();

        pt = unproject(x, y, zrange(1));
        float dist_e = pt.norm() * param.inverse_voxel_size();
        Vec3f block_e = pose * (pt)*param.inverse_voxel_size();

        Vec3f dir = (block_e - block_s).normalized();
        Vec3f result = block_s;

        bool valid_sdf = false;
        bool found_pt = false;
        float step;

        while (dist_s < dist_e)
        {
            last_sdf = sdf;
            sdf = read_sdf(result, valid_sdf);

            if (sdf <= 0.5f && sdf >= -0.5f)
                sdf = read_sdf_interped(result, valid_sdf);

            if (sdf <= 0.0f)
                break;

            if (sdf >= 0.f && last_sdf < 0.f)
                return;

            if (valid_sdf)
                step = max(sdf * param.raycast_step_scale(), 1.0f);
            else
                step = 2;

            result += step * dir;
            dist_s += step;
        }

        if (sdf <= 0.0f)
        {
            step = sdf * param.raycast_step_scale();
            result += step * dir;

            sdf = read_sdf_interped(result, valid_sdf);

            step = sdf * param.raycast_step_scale();
            result += step * dir;

            if (valid_sdf)
                found_pt = true;
        }

        if (found_pt)
        {
            result = inv_pose * (result * param.voxel_size);
            vmap.ptr(y)[x] = Vec4f(result(0), result(1), result(2), 1.0);
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

void raycast(MapStorage map_struct,
             MapState state,
             cv::cuda::GpuMat vmap,
             cv::cuda::GpuMat nmap,
             cv::cuda::GpuMat zrange_x,
             cv::cuda::GpuMat zrange_y,
             const Sophus::SE3d &pose,
             const Eigen::Matrix3d &intrinsic_matrix)
{
    const int cols = vmap.cols;
    const int rows = vmap.rows;

    MapRenderingDelegate delegate;

    delegate.width = cols;
    delegate.height = rows;
    delegate.map_struct = map_struct;
    delegate.vmap = vmap;
    delegate.nmap = nmap;
    delegate.zrange_x = zrange_x;
    delegate.zrange_y = zrange_y;
    delegate.invfx = 1.0 / intrinsic_matrix(0, 0);
    delegate.invfy = 1.0 / intrinsic_matrix(1, 1);
    delegate.cx = intrinsic_matrix(0, 2);
    delegate.cy = intrinsic_matrix(1, 2);
    delegate.pose = pose.cast<float>();
    delegate.inv_pose = pose.inverse().cast<float>();

    dim3 thread(4, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    callDeviceFunctor<<<block, thread>>>(delegate);
}

// void raycast_with_colour(MapStorage map_struct,
//                          MapState state,
//                          cv::cuda::GpuMat vmap,
//                          cv::cuda::GpuMat nmap,
//                          cv::cuda::GpuMat image,
//                          cv::cuda::GpuMat zrange_x,
//                          cv::cuda::GpuMat zrange_y,
//                          const Sophus::SE3d &pose,
//                          const Eigen::Matrix3d &intrinsic_matrix)
// {
//     const int cols = vmap.cols;
//     const int rows = vmap.rows;

//     MapRenderingDelegate delegate;

//     delegate.width = cols;
//     delegate.height = rows;
//     delegate.map_struct = map_struct;
//     delegate.vmap = vmap;
//     delegate.nmap = nmap;
//     delegate.image = image;
//     delegate.zrange_x = zrange_x;
//     delegate.zrange_y = zrange_y;
//     delegate.invfx = 1.0 / intrinsic_matrix(0, 0);
//     delegate.invfy = 1.0 / intrinsic_matrix(1, 1);
//     delegate.cx = intrinsic_matrix(0, 2);
//     delegate.cy = intrinsic_matrix(1, 2);
//     delegate.pose = pose.cast<float>().matrix3x4();
//     delegate.inv_pose = pose.inverse().cast<float>().matrix3x4();

//     dim3 thread(4, 8);
//     dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

//     callDeviceFunctor<<<block, thread>>>(delegate);
// }

// __device__ __forceinline__ bool is_vertex_visible(
//     const Vec3f &pt, SE3f Tinv,
//     int cols, int rows, float fx,
//     float fy, float cx, float cy)
// {
//     auto ptTransformed = Tinv * pt;
//     Vec2f pt2d = Vec2f(fx * ptTransformed(0) / ptTransformed(2) + cx, fy * ptTransformed(1) / ptTransformed(2) + cy);
//     return !(pt2d(0) < 0 || pt2d(1) < 0 ||
//              pt2d(0) > cols - 1 || pt2d(1) > rows - 1 ||
//              ptTransformed(2) < param.zmin_update ||
//              ptTransformed(2) > param.zmax_update);
// }

// __device__ __forceinline__ bool is_block_visible(
//     const Vec3i &block_pos,
//     const Matrix3x4f &inv_pose,
//     int cols, int rows, float fx,
//     float fy, float cx, float cy)
// {
//     float scale = param.block_size_metric();
// #pragma unroll
//     for (int corner = 0; corner < 8; ++corner)
//     {
//         Vec3i tmp = block_pos;
//         tmp(0) += (corner & 1) ? 1 : 0;
//         tmp(1) += (corner & 2) ? 1 : 0;
//         tmp(2) += (corner & 4) ? 1 : 0;

//         if (is_vertex_visible(tmp * scale, inv_pose, cols, rows, fx, fy, cx, cy))
//             return true;
//     }

//     return false;
// }
