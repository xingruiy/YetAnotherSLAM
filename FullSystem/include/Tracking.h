#pragma once
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "Viewer.h"
#include "LocalMapping.h"
#include "System.h"
#include "GlobalDef.h"
#include "DenseMapping.h"
#include "DenseTracking.h"

namespace SLAM
{

class Viewer;
class LocalMapping;
class System;

class Tracking
{
public:
    Tracking(System *system, Map *map, Viewer *mpViewer, LocalMapping *mpLocalMapping);
    void trackImage(cv::Mat ImGray, cv::Mat Depth, const double TimeStamp);
    void reset();

private:
    enum TrackingState
    {
        Null,
        OK,
        Lost
    };

    enum TrackingModal
    {
        RGB_ONLY,
        DEPTH_ONLY,
        RGB_AND_DEPTH,
        IDLE
    };

    Frame NextFrame;
    Frame lastFrame;
    Sophus::SE3d T_ref2World;

    void Initialisation();
    bool trackLastFrame();
    bool Relocalisation();
    bool NeedNewKeyFrame();
    void MakeNewKeyFrame();

    System *mpSystem;
    Map *mpMap;
    Viewer *mpViewer;

    DenseMapping *mpMapper;
    DenseTracking *mpTracker;
    LocalMapping *mpLocalMapping;

    TrackingState trackingState;
    TrackingModal trackingModal;

    cv::Mat mDescriptors;
    cv::cuda::GpuMat mCurrentMapPrediction;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
};

} // namespace SLAM