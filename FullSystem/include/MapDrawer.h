#pragma once
#include <mutex>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include "KeyFrame.h"
#include "MapPoint.h"
#include "VoxelMap.h"
#include "Map.h"

namespace SLAM
{

class MapDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapDrawer(Map *pMap);

    void DrawMapPoints(int iPointSize);
    void DrawKeyFrames(bool bDrawKF, bool bDrawGraph, int iEdgeWeight);

    void DrawMesh(int N = -1);

private:
    Map *mpMap;
    std::mutex mPoseMutex;
    Eigen::Matrix4f mCameraPose;

    Eigen::Matrix3f mCalibInv;
    int width, height;
};

} // namespace SLAM
