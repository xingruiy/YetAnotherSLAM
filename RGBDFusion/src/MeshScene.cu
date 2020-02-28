#include "ParallelScan.h"
#include "CudaUtils.h"
#include "TriangleTable.h"
#include "VoxelMapping.h"
#include "MappingUtils.h"
#include "VoxelStructUtils.h"

#define MAX_NUM_MESH_TRIANGLES 20000000

struct MeshSceneFunctor
{
    int hashTableSize;
    int bucketSize;
    float voxelSize;
    size_t bufferSize;

    HashEntry *mplEntry;
    HashEntry *mplHashTable;
    Voxel *mplVoxelBlocks;

    uint *mpNumEntry;
    uint *mpNumTriangle;

    Eigen::Vector3f *mplNormals;
    Eigen::Vector3f *mplTriangle;

    __device__ __forceinline__ void SelectBlocksToRender() const
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        __shared__ bool needScan;

        if (x == 0)
            needScan = false;

        __syncthreads();
        uint val = 0;
        if (x < hashTableSize && mplHashTable[x].ptr >= 0)
        {
            needScan = true;
            val = 1;
        }

        __syncthreads();
        if (needScan)
        {
            int offset = ParallelScan<1024>(val, mpNumEntry);
            if (offset != -1)
                mplEntry[offset] = mplHashTable[x];
        }
    }

    __device__ __forceinline__ float read_sdf(Eigen::Vector3f pt, bool &valid) const
    {
        Voxel *voxel = NULL;
        findVoxel(floor(pt), voxel, mplHashTable, mplVoxelBlocks, bucketSize);
        if (voxel && voxel->wt != 0)
        {
            valid = true;
            return UnPackFloat(voxel->sdf);
        }
        else
        {
            valid = false;
            return 1.0f;
        }
    }

    __device__ __forceinline__ bool read_sdf_list(float *sdf, Eigen::Vector3f pos) const
    {
        bool valid = false;
        sdf[0] = read_sdf(pos, valid);
        if (!valid)
            return false;

        sdf[1] = read_sdf(pos + Eigen::Vector3f(1, 0, 0), valid);
        if (!valid)
            return false;

        sdf[2] = read_sdf(pos + Eigen::Vector3f(1, 1, 0), valid);
        if (!valid)
            return false;

        sdf[3] = read_sdf(pos + Eigen::Vector3f(0, 1, 0), valid);
        if (!valid)
            return false;

        sdf[4] = read_sdf(pos + Eigen::Vector3f(0, 0, 1), valid);
        if (!valid)
            return false;

        sdf[5] = read_sdf(pos + Eigen::Vector3f(1, 0, 1), valid);
        if (!valid)
            return false;

        sdf[6] = read_sdf(pos + Eigen::Vector3f(1, 1, 1), valid);
        if (!valid)
            return false;

        sdf[7] = read_sdf(pos + Eigen::Vector3f(0, 1, 1), valid);
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

    __device__ __forceinline__ int make_vertex(Eigen::Vector3f *verts, const Eigen::Vector3f &pos) const
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
            float val = interpolateLinear(sdf[0], sdf[1]);
            verts[0] = pos + Eigen::Vector3f(val, 0, 0);
        }
        if (edgeTable[cubeIdx] & 2)
        {
            float val = interpolateLinear(sdf[1], sdf[2]);
            verts[1] = pos + Eigen::Vector3f(1, val, 0);
        }
        if (edgeTable[cubeIdx] & 4)
        {
            float val = interpolateLinear(sdf[2], sdf[3]);
            verts[2] = pos + Eigen::Vector3f(1 - val, 1, 0);
        }
        if (edgeTable[cubeIdx] & 8)
        {
            float val = interpolateLinear(sdf[3], sdf[0]);
            verts[3] = pos + Eigen::Vector3f(0, 1 - val, 0);
        }
        if (edgeTable[cubeIdx] & 16)
        {
            float val = interpolateLinear(sdf[4], sdf[5]);
            verts[4] = pos + Eigen::Vector3f(val, 0, 1);
        }
        if (edgeTable[cubeIdx] & 32)
        {
            float val = interpolateLinear(sdf[5], sdf[6]);
            verts[5] = pos + Eigen::Vector3f(1, val, 1);
        }
        if (edgeTable[cubeIdx] & 64)
        {
            float val = interpolateLinear(sdf[6], sdf[7]);
            verts[6] = pos + Eigen::Vector3f(1 - val, 1, 1);
        }
        if (edgeTable[cubeIdx] & 128)
        {
            float val = interpolateLinear(sdf[7], sdf[4]);
            verts[7] = pos + Eigen::Vector3f(0, 1 - val, 1);
        }
        if (edgeTable[cubeIdx] & 256)
        {
            float val = interpolateLinear(sdf[0], sdf[4]);
            verts[8] = pos + Eigen::Vector3f(0, 0, val);
        }
        if (edgeTable[cubeIdx] & 512)
        {
            float val = interpolateLinear(sdf[1], sdf[5]);
            verts[9] = pos + Eigen::Vector3f(1, 0, val);
        }
        if (edgeTable[cubeIdx] & 1024)
        {
            float val = interpolateLinear(sdf[2], sdf[6]);
            verts[10] = pos + Eigen::Vector3f(1, 1, val);
        }
        if (edgeTable[cubeIdx] & 2048)
        {
            float val = interpolateLinear(sdf[3], sdf[7]);
            verts[11] = pos + Eigen::Vector3f(0, 1, val);
        }

        return cubeIdx;
    }

    __device__ __forceinline__ void operator()() const
    {
        int x = blockIdx.y * gridDim.x + blockIdx.x;
        if (*mpNumTriangle >= bufferSize || x >= *mpNumEntry)
            return;

        Eigen::Vector3f verts[12];
        Eigen::Vector3i pos = mplEntry[x].pos * BlockSize;

        for (int voxelIdxZ = 0; voxelIdxZ < BlockSize; ++voxelIdxZ)
        {
            Eigen::Vector3i localPos = Eigen::Vector3i(threadIdx.x, threadIdx.y, voxelIdxZ);
            int cubeIdx = make_vertex(verts, (pos + localPos).cast<float>());
            if (cubeIdx <= 0)
                continue;

            for (int i = 0; triTable[cubeIdx][i] != -1; i += 3)
            {
                uint triangleId = atomicAdd(mpNumTriangle, 1);
                if (triangleId < bufferSize)
                {
                    mplTriangle[triangleId * 3] = verts[triTable[cubeIdx][i]] * voxelSize;
                    mplTriangle[triangleId * 3 + 1] = verts[triTable[cubeIdx][i + 1]] * voxelSize;
                    mplTriangle[triangleId * 3 + 2] = verts[triTable[cubeIdx][i + 2]] * voxelSize;
                    auto v10 = mplTriangle[triangleId * 3 + 1] - mplTriangle[triangleId * 3];
                    auto v20 = mplTriangle[triangleId * 3 + 2] - mplTriangle[triangleId * 3];
                    auto n = v10.cross(v20).normalized();
                    mplNormals[triangleId * 3] = n;
                    mplNormals[triangleId * 3 + 1] = n;
                    mplNormals[triangleId * 3 + 2] = n;
                }
            }
        }
    }
};

__global__ void selectBlockKernel(MeshSceneFunctor bva)
{
    bva.SelectBlocksToRender();
}

void create_mesh_with_normal(
    MapStruct map_struct,
    uint &mpNumEntry,
    uint &mpNumTriangle,
    void *vertexBuffer,
    void *normalBuffer,
    size_t bufferSize)
{
    uint *cuda_block_count;
    uint *cuda_triangle_count;
    cudaMalloc(&cuda_block_count, sizeof(uint));
    cudaMalloc(&cuda_triangle_count, sizeof(uint));
    cudaMemset(cuda_block_count, 0, sizeof(uint));
    cudaMemset(cuda_triangle_count, 0, sizeof(uint));

    MeshSceneFunctor bva;
    bva.mplEntry = map_struct.visibleTable;
    bva.mpNumEntry = cuda_block_count;
    bva.mpNumTriangle = cuda_triangle_count;
    bva.mplTriangle = static_cast<Eigen::Vector3f *>(vertexBuffer);
    bva.mplNormals = static_cast<Eigen::Vector3f *>(normalBuffer);
    bva.mplHashTable = map_struct.mplHashTable;
    bva.mplVoxelBlocks = map_struct.mplVoxelBlocks;
    bva.hashTableSize = map_struct.hashTableSize;
    bva.bucketSize = map_struct.bucketSize;
    bva.voxelSize = map_struct.voxelSize;
    bva.bufferSize = MAX_NUM_MESH_TRIANGLES;

    dim3 thread(1024);
    dim3 block = dim3(cv::divUp(map_struct.hashTableSize, thread.x));

    selectBlockKernel<<<block, thread>>>(bva);

    (cudaMemcpy(&mpNumEntry, cuda_block_count, sizeof(uint), cudaMemcpyDeviceToHost));
    if (mpNumEntry == 0)
        return;

    thread = dim3(8, 8);
    block = dim3(cv::divUp((size_t)mpNumEntry, 16U), 16U);

    callDeviceFunctor<<<block, thread>>>(bva);

    cudaMemcpy(&mpNumTriangle, cuda_triangle_count, sizeof(uint), cudaMemcpyDeviceToHost);
    mpNumTriangle = std::min(mpNumTriangle, (uint)MAX_NUM_MESH_TRIANGLES);

    cudaFree(cuda_block_count);
    cudaFree(cuda_triangle_count);
}