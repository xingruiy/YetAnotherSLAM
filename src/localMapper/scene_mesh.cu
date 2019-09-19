#include "map_proc.h"
#include "utils/prefixSum.h"
#include "utils/cudaUtils.h"
#include "utils/triangleTable.h"
#include "localMapper/localMapper.h"

#define MAX_NUM_MESH_TRIANGLES 20000000

struct BuildVertexArray
{
    Vec3f *triangles;
    HashEntry *block_array;
    uint *block_count;
    uint *triangle_count;
    Vec3f *surfaceNormal;

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
        if (x < hashTableSize && hashTable[x].ptr >= 0)
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
        if (voxel && voxel->wt != 0)
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

    __device__ __forceinline__ bool read_sdf_list(float *sdf, Vec3f pos) const
    {
        bool valid = false;
        sdf[0] = read_sdf(pos, valid);
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

    __device__ __forceinline__ int make_vertex(Vec3f *verts, const Vec3f &pos) const
    {
        float sdf[8];

        if (!read_sdf_list(sdf, pos))
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
            float val = interpolate_sdf(sdf[0], sdf[1]);
            verts[0] = pos + Vec3f(val, 0, 0);
        }
        if (edgeTable[cubeIdx] & 2)
        {
            float val = interpolate_sdf(sdf[1], sdf[2]);
            verts[1] = pos + Vec3f(1, val, 0);
        }
        if (edgeTable[cubeIdx] & 4)
        {
            float val = interpolate_sdf(sdf[2], sdf[3]);
            verts[2] = pos + Vec3f(1 - val, 1, 0);
        }
        if (edgeTable[cubeIdx] & 8)
        {
            float val = interpolate_sdf(sdf[3], sdf[0]);
            verts[3] = pos + Vec3f(0, 1 - val, 0);
        }
        if (edgeTable[cubeIdx] & 16)
        {
            float val = interpolate_sdf(sdf[4], sdf[5]);
            verts[4] = pos + Vec3f(val, 0, 1);
        }
        if (edgeTable[cubeIdx] & 32)
        {
            float val = interpolate_sdf(sdf[5], sdf[6]);
            verts[5] = pos + Vec3f(1, val, 1);
        }
        if (edgeTable[cubeIdx] & 64)
        {
            float val = interpolate_sdf(sdf[6], sdf[7]);
            verts[6] = pos + Vec3f(1 - val, 1, 1);
        }
        if (edgeTable[cubeIdx] & 128)
        {
            float val = interpolate_sdf(sdf[7], sdf[4]);
            verts[7] = pos + Vec3f(0, 1 - val, 1);
        }
        if (edgeTable[cubeIdx] & 256)
        {
            float val = interpolate_sdf(sdf[0], sdf[4]);
            verts[8] = pos + Vec3f(0, 0, val);
        }
        if (edgeTable[cubeIdx] & 512)
        {
            float val = interpolate_sdf(sdf[1], sdf[5]);
            verts[9] = pos + Vec3f(1, 0, val);
        }
        if (edgeTable[cubeIdx] & 1024)
        {
            float val = interpolate_sdf(sdf[2], sdf[6]);
            verts[10] = pos + Vec3f(1, 1, val);
        }
        if (edgeTable[cubeIdx] & 2048)
        {
            float val = interpolate_sdf(sdf[3], sdf[7]);
            verts[11] = pos + Vec3f(0, 1, val);
        }

        return cubeIdx;
    }

    __device__ __forceinline__ void operator()() const
    {
        int x = blockIdx.y * gridDim.x + blockIdx.x;
        if (*triangle_count >= MAX_NUM_MESH_TRIANGLES || x >= *block_count)
            return;

        Vec3f verts[12];
        Vec3i pos = block_array[x].pos * BlockSize;

        for (int voxelIdxZ = 0; voxelIdxZ < BlockSize; ++voxelIdxZ)
        {
            Vec3i localPos = Vec3i(threadIdx.x, threadIdx.y, voxelIdxZ);
            int cubeIdx = make_vertex(verts, (pos + localPos).cast<float>());
            if (cubeIdx <= 0)
                continue;

            for (int i = 0; triTable[cubeIdx][i] != -1; i += 3)
            {
                uint triangleId = atomicAdd(triangle_count, 1);
                if (triangleId < MAX_NUM_MESH_TRIANGLES)
                {
                    triangles[triangleId * 3] = verts[triTable[cubeIdx][i]] * voxelSize;
                    triangles[triangleId * 3 + 1] = verts[triTable[cubeIdx][i + 1]] * voxelSize;
                    triangles[triangleId * 3 + 2] = verts[triTable[cubeIdx][i + 2]] * voxelSize;
                    auto v10 = triangles[triangleId * 3 + 1] - triangles[triangleId * 3];
                    auto v20 = triangles[triangleId * 3 + 2] - triangles[triangleId * 3];
                    auto n = v10.cross(v20).normalized();
                    surfaceNormal[triangleId * 3] = n;
                    surfaceNormal[triangleId * 3 + 1] = n;
                    surfaceNormal[triangleId * 3 + 2] = n;
                }
            }
        }
    }
};

__global__ void selectBlockKernel(BuildVertexArray bva)
{
    bva.select_blocks();
}

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
    bva.surfaceNormal = static_cast<Vec3f *>(vertex_normal);
    bva.hashTable = map_struct.hash_table_;
    bva.listBlocks = map_struct.voxels_;
    bva.hashTableSize = map_struct.hashTableSize;
    bva.bucketSize = map_struct.bucketSize;
    bva.voxelSize = map_struct.voxelSize;

    dim3 thread(1024);
    dim3 block = dim3(div_up(map_struct.hashTableSize, thread.x));

    selectBlockKernel<<<block, thread>>>(bva);

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