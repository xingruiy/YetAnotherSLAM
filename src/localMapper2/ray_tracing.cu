#include "map_struct.h"
#include "matrix_type.h"
#include "vector_type.h"
#include "safe_call.h"
#include "utils/prefixSum.h"
#include <opencv2/opencv.hpp>

#define RENDERING_BLOCK_SIZE_X 16
#define RENDERING_BLOCK_SIZE_Y 16
#define RENDERING_BLOCK_SUBSAMPLE 8

struct RenderingBlockDelegate
{
    int width, height;
    Matrix3x4f inv_pose;
    float fx, fy, cx, cy;

    uint *rendering_block_count;
    uint visible_block_count;

    HashEntry *visible_block_pos;
    mutable cv::cuda::PtrStepSz<float> zrange_x;
    mutable cv::cuda::PtrStep<float> zrange_y;
    RenderingBlock *rendering_blocks;

    __device__ inline Vector2f project(const Vector3f &pt) const
    {
        return Vector2f(fx * pt.x / pt.z + cx, fy * pt.y / pt.z + cy);
    }

    // compare val with the old value stored in *add
    // and write the bigger one to *add
    __device__ inline void atomic_max(float *add, float val) const
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
    __device__ inline void atomic_min(float *add, float val) const
    {
        int *address_as_i = (int *)add;
        int old = *address_as_i, assumed;
        do
        {
            assumed = old;
            old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }

    __device__ inline bool create_rendering_block(const Vector3i &block_pos, RenderingBlock &block) const
    {
        block.upper_left = Vector2s(zrange_x.cols, zrange_x.rows);
        block.lower_right = Vector2s(-1, -1);
        block.zrange = Vector2f(param.zmax_raycast, param.zmin_raycast);

#pragma unroll
        for (int corner = 0; corner < 8; ++corner)
        {
            Vector3i tmp = block_pos;
            tmp.x += (corner & 1) ? 1 : 0;
            tmp.y += (corner & 2) ? 1 : 0;
            tmp.z += (corner & 4) ? 1 : 0;

            Vector3f pt3d = tmp * param.block_size_metric();
            pt3d = inv_pose(pt3d);

            Vector2f pt2d = project(pt3d) / RENDERING_BLOCK_SUBSAMPLE;

            if (block.upper_left.x > std::floor(pt2d.x))
                block.upper_left.x = (int)std::floor(pt2d.x);

            if (block.lower_right.x < ceil(pt2d.x))
                block.lower_right.x = (int)ceil(pt2d.x);

            if (block.upper_left.y > std::floor(pt2d.y))
                block.upper_left.y = (int)std::floor(pt2d.y);

            if (block.lower_right.y < ceil(pt2d.y))
                block.lower_right.y = (int)ceil(pt2d.y);

            if (block.zrange.x > pt3d.z)
                block.zrange.x = pt3d.z;

            if (block.zrange.y < pt3d.z)
                block.zrange.y = pt3d.z;
        }

        if (block.upper_left.x < 0)
            block.upper_left.x = 0;

        if (block.upper_left.y < 0)
            block.upper_left.y = 0;

        if (block.lower_right.x >= zrange_x.cols)
            block.lower_right.x = zrange_x.cols - 1;

        if (block.lower_right.y >= zrange_x.rows)
            block.lower_right.y = zrange_x.rows - 1;

        if (block.upper_left.x > block.lower_right.x)
            return false;

        if (block.upper_left.y > block.lower_right.y)
            return false;

        if (block.zrange.x < param.zmin_raycast)
            block.zrange.x = param.zmin_raycast;

        if (block.zrange.y < param.zmin_raycast)
            return false;

        return true;
    }

    __device__ inline void create_rendering_block_list(int offset, const RenderingBlock &block, int &nx, int &ny) const
    {
        for (int y = 0; y < ny; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                if (offset < param.num_max_rendering_blocks_)
                {
                    RenderingBlock &b(rendering_blocks[offset++]);
                    b.upper_left.x = block.upper_left.x + x * RENDERING_BLOCK_SIZE_X;
                    b.upper_left.y = block.upper_left.y + y * RENDERING_BLOCK_SIZE_Y;
                    b.lower_right.x = block.upper_left.x + (x + 1) * RENDERING_BLOCK_SIZE_X;
                    b.lower_right.y = block.upper_left.y + (y + 1) * RENDERING_BLOCK_SIZE_Y;

                    if (b.lower_right.x > block.lower_right.x)
                        b.lower_right.x = block.lower_right.x;

                    if (b.lower_right.y > block.lower_right.y)
                        b.lower_right.y = block.lower_right.y;

                    b.zrange = block.zrange;
                }
            }
        }
    }

    __device__ inline void operator()() const
    {
        int x = threadIdx.x + blockDim.x * blockIdx.x;

        bool valid = false;
        uint requiredNoBlocks = 0;
        RenderingBlock block;
        int nx, ny;

        if (x < visible_block_count && visible_block_pos[x].ptr_ != -1)
        {
            valid = create_rendering_block(visible_block_pos[x].pos_, block);
            float dx = (float)block.lower_right.x - block.upper_left.x + 1;
            float dy = (float)block.lower_right.y - block.upper_left.y + 1;
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

    __device__ inline void fill_rendering_blocks() const
    {
        int x = threadIdx.x;
        int y = threadIdx.y;

        int block = blockIdx.x * 4 + blockIdx.y;
        if (block >= param.num_max_rendering_blocks_)
            return;

        RenderingBlock &b(rendering_blocks[block]);

        int xpos = b.upper_left.x + x;
        if (xpos > b.lower_right.x || xpos >= zrange_x.cols)
            return;

        int ypos = b.upper_left.y + y;
        if (ypos > b.lower_right.y || ypos >= zrange_x.rows)
            return;

        atomic_min(&zrange_x.ptr(ypos)[xpos], b.zrange.x);
        atomic_max(&zrange_y.ptr(ypos)[xpos], b.zrange.y);

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
    safe_call(cudaMalloc((void **)&count_device, sizeof(uint)));
    safe_call(cudaMemset((void *)count_device, 0, sizeof(uint)));

    RenderingBlockDelegate delegate;

    delegate.width = cols;
    delegate.height = rows;
    delegate.inv_pose = frame_pose.inverse().cast<float>().matrix3x4();
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

    call_device_functor<<<block, thread>>>(delegate);

    safe_call(cudaMemcpy(&count_rendering_block, count_device, sizeof(uint), cudaMemcpyDeviceToHost));
    if (count_rendering_block == 0)
        return;

    thread = dim3(RENDERING_BLOCK_SIZE_X, RENDERING_BLOCK_SIZE_Y);
    block = dim3((uint)ceil((float)count_rendering_block / 4), 4);

    split_and_fill_rendering_blocks_kernel<<<block, thread>>>(delegate);
    safe_call(cudaFree((void *)count_device));
}

struct MapRenderingDelegate
{
    int width, height;
    MapStorage map_struct;
    mutable cv::cuda::PtrStep<Vector4f> vmap;
    mutable cv::cuda::PtrStep<Vector4f> nmap;
    mutable cv::cuda::PtrStep<Vector3c> image;
    cv::cuda::PtrStepSz<float> zrange_x;
    cv::cuda::PtrStepSz<float> zrange_y;
    float invfx, invfy, cx, cy;
    Matrix3x4f pose, inv_pose;

    __device__ inline float read_sdf(const Vector3f &pt3d, bool &valid)
    {
        Voxel *voxel = NULL;
        findVoxel(map_struct, ToVector3i(pt3d), voxel);
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

    __device__ inline float read_sdf_interped(const Vector3f &pt, bool &valid)
    {
        Vector3f xyz = pt - floor(pt);
        float sdf[2], result[4];
        bool valid_pt;

        sdf[0] = read_sdf(pt, valid_pt);
        sdf[1] = read_sdf(pt + Vector3f(1, 0, 0), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];

        sdf[0] = read_sdf(pt + Vector3f(0, 1, 0), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Vector3f(1, 1, 0), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];
        result[2] = (1.0f - xyz.y) * result[0] + xyz.y * result[1];

        sdf[0] = read_sdf(pt + Vector3f(0, 0, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Vector3f(1, 0, 1), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];

        sdf[0] = read_sdf(pt + Vector3f(0, 1, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Vector3f(1, 1, 1), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];
        result[3] = (1.0f - xyz.y) * result[0] + xyz.y * result[1];
        valid = valid_pt;
        return (1.0f - xyz.z) * result[2] + xyz.z * result[3];
    }

    __device__ inline Vector3f unproject(const int &x, const int &y, const float &z) const
    {
        return Vector3f((x - cx) * invfx * z, (y - cy) * invfy * z, z);
    }

    __device__ inline void operator()()
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= width || y >= height)
            return;

        vmap.ptr(y)[x] = Vector4f(__int_as_float(0x7fffffff));

        Vector2i local_id;
        local_id.x = __float2int_rd((float)x / 8);
        local_id.y = __float2int_rd((float)y / 8);

        Vector2f zrange;
        zrange.x = zrange_x.ptr(local_id.y)[local_id.x];
        zrange.y = zrange_y.ptr(local_id.y)[local_id.x];
        if (zrange.y < 1e-3 || zrange.x < 1e-3 || isnan(zrange.x) || isnan(zrange.y))
            return;

        float sdf = 1.0f;
        float last_sdf;

        Vector3f pt = unproject(x, y, zrange.x);
        float dist_s = pt.norm() * param.inverse_voxel_size();
        Vector3f block_s = pose(pt) * param.inverse_voxel_size();

        pt = unproject(x, y, zrange.y);
        float dist_e = pt.norm() * param.inverse_voxel_size();
        Vector3f block_e = pose(pt) * param.inverse_voxel_size();

        Vector3f dir = normalised(block_e - block_s);
        Vector3f result = block_s;

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
            result = inv_pose(result * param.voxel_size);
            vmap.ptr(y)[x] = Vector4f(result.x, result.y, result.z, 1.0);
        }
    }

    __device__ inline Vector3c read_colour(Vector3f pt3d, bool &valid)
    {
        Voxel *voxel = NULL;
        findVoxel(map_struct, ToVector3i(pt3d), voxel);
        if (voxel && voxel->weight != 0)
        {
            valid = true;
            return voxel->rgb;
        }
        else
        {
            valid = false;
            return Vector3c(0);
        }
    }

    __device__ inline Vector3c read_colour_interpolated(Vector3f pt, bool &valid)
    {
        Vector3f xyz = pt - floor(pt);
        Vector3c sdf[2];
        Vector3f result[4];
        bool valid_pt;

        sdf[0] = read_colour(pt, valid_pt);
        sdf[1] = read_colour(pt + Vector3f(1, 0, 0), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];

        sdf[0] = read_colour(pt + Vector3f(0, 1, 0), valid);
        valid_pt &= valid;
        sdf[1] = read_colour(pt + Vector3f(1, 1, 0), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];
        result[2] = (1.0f - xyz.y) * result[0] + xyz.y * result[1];

        sdf[0] = read_colour(pt + Vector3f(0, 0, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_colour(pt + Vector3f(1, 0, 1), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];

        sdf[0] = read_colour(pt + Vector3f(0, 1, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_colour(pt + Vector3f(1, 1, 1), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];
        result[3] = (1.0f - xyz.y) * result[0] + xyz.y * result[1];
        valid = valid_pt;
        return ToVector3c((1.0f - xyz.z) * result[2] + xyz.z * result[3]);
    }

    __device__ inline void raycast_with_colour()
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= width || y >= height)
            return;

        vmap.ptr(y)[x] = Vector4f(__int_as_float(0x7fffffff));
        image.ptr(y)[x] = Vector3c(255);

        Vector2i local_id;
        local_id.x = __float2int_rd((float)x / 8);
        local_id.y = __float2int_rd((float)y / 8);

        Vector2f zrange;
        zrange.x = zrange_x.ptr(local_id.y)[local_id.x];
        zrange.y = zrange_y.ptr(local_id.y)[local_id.x];
        if (zrange.y < 1e-3 || zrange.x < 1e-3 || isnan(zrange.x) || isnan(zrange.y))
            return;

        float sdf = 1.0f;
        float last_sdf;

        Vector3f pt = unproject(x, y, zrange.x);
        float dist_s = pt.norm() * param.inverse_voxel_size();
        Vector3f block_s = pose(pt) * param.inverse_voxel_size();

        pt = unproject(x, y, zrange.y);
        float dist_e = pt.norm() * param.inverse_voxel_size();
        Vector3f block_e = pose(pt) * param.inverse_voxel_size();

        Vector3f dir = normalised(block_e - block_s);
        Vector3f result = block_s;

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
            // auto rgb = read_colour_interpolated(result, valid_sdf);
            auto rgb = read_colour(result, valid_sdf);
            if (!valid_sdf)
                return;

            result = inv_pose(result * param.voxel_size);
            vmap.ptr(y)[x] = Vector4f(result.x, result.y, result.z, 1.0f);
            image.ptr(y)[x] = rgb;
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
    delegate.pose = pose.cast<float>().matrix3x4();
    delegate.inv_pose = pose.inverse().cast<float>().matrix3x4();

    dim3 thread(4, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    call_device_functor<<<block, thread>>>(delegate);
}

void raycast_with_colour(MapStorage map_struct,
                         MapState state,
                         cv::cuda::GpuMat vmap,
                         cv::cuda::GpuMat nmap,
                         cv::cuda::GpuMat image,
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
    delegate.image = image;
    delegate.zrange_x = zrange_x;
    delegate.zrange_y = zrange_y;
    delegate.invfx = 1.0 / intrinsic_matrix(0, 0);
    delegate.invfy = 1.0 / intrinsic_matrix(1, 1);
    delegate.cx = intrinsic_matrix(0, 2);
    delegate.cy = intrinsic_matrix(1, 2);
    delegate.pose = pose.cast<float>().matrix3x4();
    delegate.inv_pose = pose.inverse().cast<float>().matrix3x4();

    dim3 thread(4, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    call_device_functor<<<block, thread>>>(delegate);
}

__device__ inline bool is_vertex_visible(
    Vector3f pt, Matrix3x4f inv_pose,
    int cols, int rows, float fx,
    float fy, float cx, float cy)
{
    pt = inv_pose(pt);
    Vector2f pt2d = Vector2f(fx * pt.x / pt.z + cx, fy * pt.y / pt.z + cy);
    return !(pt2d.x < 0 || pt2d.y < 0 ||
             pt2d.x > cols - 1 || pt2d.y > rows - 1 ||
             pt.z < param.zmin_update || pt.z > param.zmax_update);
}

__device__ inline bool is_block_visible(
    const Vector3i &block_pos,
    const Matrix3x4f &inv_pose,
    int cols, int rows, float fx,
    float fy, float cx, float cy)
{
    float scale = param.block_size_metric();
#pragma unroll
    for (int corner = 0; corner < 8; ++corner)
    {
        Vector3i tmp = block_pos;
        tmp.x += (corner & 1) ? 1 : 0;
        tmp.y += (corner & 2) ? 1 : 0;
        tmp.z += (corner & 4) ? 1 : 0;

        if (is_vertex_visible(tmp * scale, inv_pose, cols, rows, fx, fy, cx, cy))
            return true;
    }

    return false;
}
