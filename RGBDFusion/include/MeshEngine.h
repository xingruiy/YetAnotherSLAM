#ifndef MESHIFICATION_H
#define MESHIFICATION_H

#include "VoxelMap.h"

class MapStruct;

class MeshEngine
{
public:
    ~MeshEngine();
    MeshEngine(int MaxNumTriangle);
    void Meshify(MapStruct *pMapStruct);

private:
    void ResetTemporaryValues();

    int mMaxNumTriangle;
    float *mplVertexBuffer;
    float *mplNormalBuffer;

    uint *cuda_block_count;
    uint *cuda_triangle_count;
};

#endif
