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

    void DrawMesh(int N, const pangolin::OpenGlMatrix &mvpMat);
    void LinkGlSlProgram();

private:
    Map *mpMap;
    std::mutex mmMutexPose;
    Eigen::Matrix4f mCameraPose;

    Eigen::Matrix3f mCalibInv;
    int width, height;

    pangolin::GlSlProgram mShader;
};

} // namespace SLAM
