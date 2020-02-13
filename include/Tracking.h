#pragma once
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "Frame.h"
#include "Viewer.h"
#include "Mapping.h"
#include "System.h"
#include "GlobalDef.h"
#include "DenseMapping.h"
#include "DenseTracking.h"

namespace SLAM
{

class Viewer;
class Mapping;
class System;

class Tracking
{
public:
    Tracking(System *system, Map *map, Viewer *viewer, Mapping *mapping);
    void trackImage(cv::Mat ImGray, cv::Mat Depth, const double TimeStamp);
    void reset();

private:
    enum TrackingState
    {
        Null,
        OK,
        Lost
    };

    Frame NextFrame;
    Frame lastFrame;
    Sophus::SE3d T_ref2World;

    void initialisation();
    bool trackLastFrame();
    bool relocalisation();
    bool NeedNewKeyFrame();
    void MakeNewKeyFrame();

    System *slamSystem;
    Map *mpMap;
    Viewer *viewer;
    TrackingState trackingState;
    DenseTracking *tracker;
    Mapping *mapping;

    cv::cuda::GpuMat mvCurrentDepth[NUM_PYR];
    cv::cuda::GpuMat mvReferenceDepth[NUM_PYR];
    cv::cuda::GpuMat mvCurrentIntensity[NUM_PYR];
    cv::cuda::GpuMat mvReferenceIntensity[NUM_PYR];
    cv::cuda::GpuMat mvIntensityGradientX[NUM_PYR];
    cv::cuda::GpuMat mvIntensityGradientY[NUM_PYR];
    cv::cuda::GpuMat mvReferencePointTransformed[NUM_PYR];
    cv::cuda::GpuMat mvCurrentInvDepth[NUM_PYR];
    cv::cuda::GpuMat mvReferenceInvDepth[NUM_PYR];
    cv::cuda::GpuMat mvInvDepthGradientX[NUM_PYR];
    cv::cuda::GpuMat mvInvDepthGradientY[NUM_PYR];

    // GPU buffer for temporary data
    cv::cuda::GpuMat mGpuBufferFloat96x29;
    cv::cuda::GpuMat mGpuBufferFloat96x3;
    cv::cuda::GpuMat mGpuBufferFloat96x2;
    cv::cuda::GpuMat mGpuBufferFloat96x1;
    cv::cuda::GpuMat mGpuBufferFloat1x29;
    cv::cuda::GpuMat mGpuBufferFloat1x3;
    cv::cuda::GpuMat mGpuBufferFloat1x2;
    cv::cuda::GpuMat mGpuBufferFloat1x1;
    cv::cuda::GpuMat mGpuBufferVector4HxW;
    cv::cuda::GpuMat mGpuBufferVector7HxW;
};

} // namespace SLAM