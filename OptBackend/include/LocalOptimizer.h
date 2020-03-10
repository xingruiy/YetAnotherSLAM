#ifndef LOCAL_OPTIMIZER_H
#define LOCAL_OPTIMIZER_H

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "NumType.h"

class LocalOptimizer
{
public:
    LocalOptimizer(int w, int h);
    void LineariseAll();

private:
    // Host
    FrameShell frames[NUM_LOCAL_KF];

    // Device
    FramePoint *activePoints;
    Sophus::SE3f *framePoseMatrix;
};

#endif