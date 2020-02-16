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

    void initialisation();
    bool trackLastFrame();
    bool relocalisation();
    bool NeedNewKeyFrame();
    void MakeNewKeyFrame();

    System *slamSystem;
    Map *mpMap;
    Viewer *viewer;
    DenseMapping *mapper;
    DenseTracking *tracker;
    Mapping *mapping;

    TrackingState trackingState;
    TrackingModal trackingModal;
};

} // namespace SLAM