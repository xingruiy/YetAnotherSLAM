#pragma once
#include <mutex>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include "KeyFrame.h"
#include "MapPoint.h"
#include "RGBDOdometry/VoxelMap.h"

namespace slam
{

class MapManager;

class MapDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapDrawer(MapManager *pMap);

    void DrawMapPoints(int iPointSize);
    void DrawKeyFrames(bool bDrawKF, bool bDrawGraph, int iEdgeWeight);

    void DrawMesh(int N, const pangolin::OpenGlMatrix &mvpMat);
    void LinkGlSlProgram();

private:
    MapManager *mpMap;
    std::mutex mmMutexPose;
    Eigen::Matrix4f mCameraPose;

    Eigen::Matrix3f mCalibInv;
    int width, height;

    pangolin::GlSlProgram mShader;
};

} // namespace slam
