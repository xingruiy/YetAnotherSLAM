#include "map_proc.h"
#include "utils/prefixSum.h"
#include "utils/cudaUtils.h"
#include "utils/triangleTable.h"
#include "localMapper2/localMapper.h"

#define MAX_NUM_MESH_TRIANGLES 20000000

struct BuildVertexArray
{
    // MapStorage map_struct;

    Vec3f *triangles;
    HashEntry *block_array;
    uint *block_count;
    uint *triangle_count;
    Vec3f *surface_normal;

    HashEntry *hashTable;
    Voxel *listBlocks;
    int hashTableSize;
    int bucketSize;
    float voxelSize;

    __device__ __forceinline__ void select_blocks() const
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        __shared__ bool needScan;

        if (x == 0)
            needScan = false;

        __syncthreads();

        uint val = 0;
        if (x < hashTableSize && hashTable[x].ptr_ >= 0)
        {
            needScan = true;
            val = 1;
        }

        __syncthreads();

        if (needScan)
        {
            int offset = computeOffset<1024>(val, block_count);
            if (offset != -1)
            {
                block_array[offset] = hashTable[x];
            }
        }
    }

    __device__ __forceinline__ float read_sdf(Vec3f pt, bool &valid) const
    {
        Voxel *voxel = NULL;
        findVoxel(floor(pt), voxel, hashTable, listBlocks, bucketSize);
        if (voxel && voxel->weight != 0)
        {
            valid = true;
            return unpackFloat(voxel->sdf);
        }
        else
        {
            valid = false;
            return 0;
        }
    }

    __device__ __forceinline__ bool read_sdf_list(float *sdf, Vec3f pos) const
    {
        bool valid = false;
        sdf[0] = read_sdf(pos + Vec3f(0, 0, 0), valid);
        if (!valid)
            return false;

        sdf[1] = read_sdf(pos + Vec3f(1, 0, 0), valid);
        if (!valid)
            return false;

        sdf[2] = read_sdf(pos + Vec3f(1, 1, 0), valid);
        if (!valid)
            return false;

        sdf[3] = read_sdf(pos + Vec3f(0, 1, 0), valid);
        if (!valid)
            return false;

        sdf[4] = read_sdf(pos + Vec3f(0, 0, 1), valid);
        if (!valid)
            return false;

        sdf[5] = read_sdf(pos + Vec3f(1, 0, 1), valid);
        if (!valid)
            return false;

        sdf[6] = read_sdf(pos + Vec3f(1, 1, 1), valid);
        if (!valid)
            return false;

        sdf[7] = read_sdf(pos + Vec3f(0, 1, 1), valid);
        if (!valid)
            return false;

        return true;
    }

    __device__ __forceinline__ float interpolate_sdf(float &v1, float &v2) const
    {
        if (fabs(0 - v1) < 1e-6)
            return 0;
        if (fabs(0 - v2) < 1e-6)
            return 1;
        if (fabs(v1 - v2) < 1e-6)
            return 0;
        return (0 - v1) / (v2 - v1);
    }

    __device__ __forceinline__ int make_vertex(Vec3f *vertex_array, const Vec3f pos) const
    {
        float sdf[8];

        if (!read_sdf_list(sdf, pos))
            return -1;

        int cube_index = 0;
        if (sdf[0] < 0)
            cube_index |= 1;
        if (sdf[1] < 0)
            cube_index |= 2;
        if (sdf[2] < 0)
            cube_index |= 4;
        if (sdf[3] < 0)
            cube_index |= 8;
        if (sdf[4] < 0)
            cube_index |= 16;
        if (sdf[5] < 0)
            cube_index |= 32;
        if (sdf[6] < 0)
            cube_index |= 64;
        if (sdf[7] < 0)
            cube_index |= 128;

        if (edgeTable[cube_index] == 0)
            return -1;

        if (edgeTable[cube_index] & 1)
        {
            float val = interpolate_sdf(sdf[0], sdf[1]);
            vertex_array[0] = pos + Vec3f(val, 0, 0);
        }
        if (edgeTable[cube_index] & 2)
        {
            float val = interpolate_sdf(sdf[1], sdf[2]);
            vertex_array[1] = pos + Vec3f(1, val, 0);
        }
        if (edgeTable[cube_index] & 4)
        {
            float val = interpolate_sdf(sdf[2], sdf[3]);
            vertex_array[2] = pos + Vec3f(1 - val, 1, 0);
        }
        if (edgeTable[cube_index] & 8)
        {
            float val = interpolate_sdf(sdf[3], sdf[0]);
            vertex_array[3] = pos + Vec3f(0, 1 - val, 0);
        }
        if (edgeTable[cube_index] & 16)
        {
            float val = interpolate_sdf(sdf[4], sdf[5]);
            vertex_array[4] = pos + Vec3f(val, 0, 1);
        }
        if (edgeTable[cube_index] & 32)
        {
            float val = interpolate_sdf(sdf[5], sdf[6]);
            vertex_array[5] = pos + Vec3f(1, val, 1);
        }
        if (edgeTable[cube_index] & 64)
        {
            float val = interpolate_sdf(sdf[6], sdf[7]);
            vertex_array[6] = pos + Vec3f(1 - val, 1, 1);
        }
        if (edgeTable[cube_index] & 128)
        {
            float val = interpolate_sdf(sdf[7], sdf[4]);
            vertex_array[7] = pos + Vec3f(0, 1 - val, 1);
        }
        if (edgeTable[cube_index] & 256)
        {
            float val = interpolate_sdf(sdf[0], sdf[4]);
            vertex_array[8] = pos + Vec3f(0, 0, val);
        }
        if (edgeTable[cube_index] & 512)
        {
            float val = interpolate_sdf(sdf[1], sdf[5]);
            vertex_array[9] = pos + Vec3f(1, 0, val);
        }
        if (edgeTable[cube_index] & 1024)
        {
            float val = interpolate_sdf(sdf[2], sdf[6]);
            vertex_array[10] = pos + Vec3f(1, 1, val);
        }
        if (edgeTable[cube_index] & 2048)
        {
            float val = interpolate_sdf(sdf[3], sdf[7]);
            vertex_array[11] = pos + Vec3f(0, 1, val);
        }

        return cube_index;
    }

    __device__ __forceinline__ void operator()() const
    {
        int x = blockIdx.y * gridDim.x + blockIdx.x;
        if (*triangle_count >= MAX_NUM_MESH_TRIANGLES || x >= *block_count)
            return;

        Vec3f vertex_array[12];
        Vec3i pos = block_array[x].pos_ * BLOCK_SIZE;

        for (int voxel_id = 0; voxel_id < BLOCK_SIZE; ++voxel_id)
        {
            Vec3i local_pos = Vec3i(threadIdx.x, threadIdx.y, voxel_id);
            int cube_index = make_vertex(vertex_array, (pos + local_pos).cast<float>());
            if (cube_index <= 0)
                continue;

            for (int i = 0; triTable[cube_index][i] != -1; i += 3)
            {
                uint triangleId = atomicAdd(triangle_count, 1);

                if (triangleId < MAX_NUM_MESH_TRIANGLES)
                {
                    triangles[triangleId * 3] = vertex_array[triTable[cube_index][i]] * voxelSize;
                    triangles[triangleId * 3 + 1] = vertex_array[triTable[cube_index][i + 1]] * voxelSize;
                    triangles[triangleId * 3 + 2] = vertex_array[triTable[cube_index][i + 2]] * voxelSize;

                    surface_normal[triangleId * 3] = ((triangles[triangleId * 3 + 1] - triangles[triangleId * 3]).cross(triangles[triangleId * 3 + 2] - triangles[triangleId * 3])).normalized();
                    surface_normal[triangleId * 3 + 1] = surface_normal[triangleId * 3 + 2] = surface_normal[triangleId * 3];
                }
            }
        }
    }
};

__global__ void select_blocks_kernel(BuildVertexArray bva)
{
    bva.select_blocks();
}

// __global__ void generate_vertex_array_kernel(BuildVertexArray bva)
// {
//     bva.operator()<false>();
// }

// void create_mesh_vertex_only(
//     MapStorage map_struct,
//     MapState state,
//     uint &block_count,
//     HashEntry *block_list,
//     uint &triangle_count,
//     void *vertex_data)
// {
//     uint *cuda_block_count;
//     uint *cuda_triangle_count;
//     (cudaMalloc(&cuda_block_count, sizeof(uint)));
//     (cudaMalloc(&cuda_triangle_count, sizeof(uint)));
//     (cudaMemset(cuda_block_count, 0, sizeof(uint)));
//     (cudaMemset(cuda_triangle_count, 0, sizeof(uint)));

//     BuildVertexArray bva;
//     bva.map_struct = map_struct;
//     bva.block_array = block_list;
//     bva.block_count = cuda_block_count;
//     bva.triangle_count = cuda_triangle_count;
//     bva.triangles = static_cast<Vec3f *>(vertex_data);

//     dim3 thread(1024);
//     dim3 block = dim3(div_up(state.num_total_hash_entries_, thread.x));

//     select_blocks_kernel<<<block, thread>>>(bva);

//     (cudaMemcpy(&block_count, cuda_block_count, sizeof(uint), cudaMemcpyDeviceToHost));
//     if (block_count == 0)
//         return;

//     thread = dim3(8, 8);
//     block = dim3(div_up(block_count, 16), 16);

//     generate_vertex_array_kernel<<<block, thread>>>(bva);

//     (cudaMemcpy(&triangle_count, cuda_triangle_count, sizeof(uint), cudaMemcpyDeviceToHost));
//     triangle_count = std::min(triangle_count, (uint)state.num_max_mesh_triangles_);

//     (cudaFree(cuda_block_count));
//     (cudaFree(cuda_triangle_count));
// }

// __global__ void generate_vertex_and_normal_array_kernel(BuildVertexArray bva)
// {
//     bva.operator()<true>();
// }

void create_mesh_with_normal(
    MapStruct map_struct,
    // MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data,
    void *vertex_normal)
{
    uint *cuda_block_count;
    uint *cuda_triangle_count;
    (cudaMalloc(&cuda_block_count, sizeof(uint)));
    (cudaMalloc(&cuda_triangle_count, sizeof(uint)));
    (cudaMemset(cuda_block_count, 0, sizeof(uint)));
    (cudaMemset(cuda_triangle_count, 0, sizeof(uint)));

    BuildVertexArray bva;
    // bva.map_struct = map_struct;
    bva.block_array = block_list;
    bva.block_count = cuda_block_count;
    bva.triangle_count = cuda_triangle_count;
    bva.triangles = static_cast<Vec3f *>(vertex_data);
    bva.surface_normal = static_cast<Vec3f *>(vertex_normal);
    bva.hashTable = map_struct.hash_table_;
    bva.listBlocks = map_struct.voxels_;
    bva.hashTableSize = map_struct.hashTableSize;
    bva.bucketSize = map_struct.bucketSize;
    bva.voxelSize = map_struct.voxelSize;

    dim3 thread(1024);
    dim3 block = dim3(div_up(map_struct.hashTableSize, thread.x));

    select_blocks_kernel<<<block, thread>>>(bva);

    (cudaMemcpy(&block_count, cuda_block_count, sizeof(uint), cudaMemcpyDeviceToHost));
    if (block_count == 0)
        return;

    thread = dim3(8, 8);
    block = dim3(div_up(block_count, 16), 16);

    callDeviceFunctor<<<block, thread>>>(bva);

    (cudaMemcpy(&triangle_count, cuda_triangle_count, sizeof(uint), cudaMemcpyDeviceToHost));
    triangle_count = std::min(triangle_count, (uint)MAX_NUM_MESH_TRIANGLES);

    (cudaFree(cuda_block_count));
    (cudaFree(cuda_triangle_count));
}

// void create_mesh_with_colour(
//     MapStorage map_struct,
//     MapState state,
//     uint &block_count,
//     HashEntry *block_list,
//     uint &triangle_count,
//     void *vertex_data,
//     void *vertex_colour)
// {
//     uint *cuda_block_count;
//     uint *cuda_triangle_count;
//     (cudaMalloc(&cuda_block_count, sizeof(uint)));
//     (cudaMalloc(&cuda_triangle_count, sizeof(uint)));
//     (cudaMemset(cuda_block_count, 0, sizeof(uint)));
//     (cudaMemset(cuda_triangle_count, 0, sizeof(uint)));

//     BuildVertexAndColourArray delegate;
//     delegate.map_struct = map_struct;
//     delegate.block_array = block_list;
//     delegate.block_count = cuda_block_count;
//     delegate.triangle_count = cuda_triangle_count;
//     delegate.triangles = static_cast<Vec3f *>(vertex_data);
//     delegate.vertex_colour = static_cast<Vec3c *>(vertex_colour);

//     dim3 thread(1024);
//     dim3 block = dim3(div_up(state.num_total_hash_entries_, thread.x));

//     select_blocks_coloured_kernel<<<block, thread>>>(delegate);

//     (cudaMemcpy(&block_count, cuda_block_count, sizeof(uint), cudaMemcpyDeviceToHost));
//     if (block_count == 0)
//         return;

//     thread = dim3(8, 8);
//     block = dim3(div_up(block_count, 16), 16);

//     generate_vertex_and_colour_array_kernel<<<block, thread>>>(delegate);

//     (cudaMemcpy(&triangle_count, cuda_triangle_count, sizeof(uint), cudaMemcpyDeviceToHost));
//     triangle_count = std::min(triangle_count, (uint)state.num_max_mesh_triangles_);

//     (cudaFree(cuda_block_count));
//     (cudaFree(cuda_triangle_count));
// }
