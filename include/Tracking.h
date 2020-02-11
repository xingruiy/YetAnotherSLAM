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
    enum class TrackingState
    {
        NotInitialized,
        OK,
        Lost
    };

    Frame *currentFrame;
    Frame *lastFrame;
    Sophus::SE3d T_ref2World;

    void initialisation();
    bool trackLastFrame();
    bool relocalisation();
    bool needNewKeyFrame();
    void addKeyFrameCandidate();

    // Local map management
    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    TrackingState mTrackingState;

    // Dense Tracker
    DenseTracking *tracker;

    // Sparse Mapping
    Mapping *mapping;

    // Dense Mapping
    DenseMapping *mpMapper;

    // Disable mapping
    bool mbOnlyTracking;

    // Frames
    Frame mCurrentFrame;
    Frame mLastFrame;

    // ORB_SLAM2::ORBextractor *mpORBextractor;
    // ORB_SLAM2::ORBVocabulary *mpORBVocabulary;

    // System
    System *mpFullSystem;

    // Map
    Map *mpMap;

    // Map Viewr
    Viewer *viewer;

    // Used for local map
    KeyFrame *mpReferenceKF;
    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;
};

} // namespace SLAM