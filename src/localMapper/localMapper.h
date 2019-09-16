#pragma once
#include <memory>
#include "localMapper/localMap.h"
#include "utils/numType.h"
#include "utils/frame.h"

class LocalMapper
{
    int frameWidth;
    int frameHeight;
    Mat33d Intrinsics;
    std::shared_ptr<VoxelMap> localMap;

public:
    LocalMapper(int w, int h, Mat33d &K);
    void fuseFrame(std::shared_ptr<Frame> frame);
    void raytracing(Mat &vmap, Mat &image);
    size_t getMesh(float *vertex, float *normal, size_t bufferSize);
};