#ifndef RAY_TRACE_ENGINE
#define RAY_TRACE_ENGINE

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "VoxelMap.h"

class MapStruct;

class RayTracer
{
public:
    ~RayTracer();
    RayTracer(int w, int h, const Eigen::Matrix3f &K);
    void RayTrace(MapStruct *pMapStruct, const Sophus::SE3d &Tcw);

    struct RenderingBlock
    {
        Eigen::Matrix<short, 2, 1> upper_left;
        Eigen::Matrix<short, 2, 1> lower_right;
        Eigen::Vector2f zrange;
    };

    uint GetNumVisibleBlock();
    uint GetNumRenderingBlocks();
    cv::cuda::GpuMat GetVMap();

protected:
    void UpdateRenderingBlocks(MapStruct *pMS, const Sophus::SE3d &Tcw);
    void Reset();

    int w, h;
    float fx, fy, cx, cy;
    float invfx, invfy;
    Eigen::Matrix3f mK;

    cv::cuda::GpuMat mTracedvmap;
    cv::cuda::GpuMat mTracedImage;

    cv::cuda::GpuMat mDepthMapMin;
    cv::cuda::GpuMat mDepthMapMax;

    uint *mpNumVisibleBlocks;
    uint *mpNumRenderingBlocks;

    RenderingBlock *mplRenderingBlockList;
};

#endif