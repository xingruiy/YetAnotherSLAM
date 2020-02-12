#pragma once
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "Frame.h"
#include "Viewer.h"
#include "Mapping.h"
#include "System.h"
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
    void trackImage(cv::Mat img, cv::Mat depth, const double timeStamp);
    void reset();

private:
    enum TrackingState
    {
        Null,
        OK,
        Lost
    };

    Frame *currentFrame;
    Frame *lastFrame;
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
};

} // namespace SLAM