#pragma once
#include <mutex>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

namespace slam
{

class Map;
class KeyFrame;
class MapPoint;
class VoxelMap;

class MapDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapDrawer(Map *mpMap);

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

} // namespace slam
