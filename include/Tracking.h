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
    Tracking(System *pSys, Map *pMap, Viewer *pViewer, Mapping *pMapping);
    void trackImage(const cv::Mat &imGray, const cv::Mat &imDepth, double TimeStamp);
    void reset();

private:
    enum class TrackingState
    {
        NotInitialized,
        OK,
        Lost
    };

    void Initialization();
    bool TrackLastFrame();
    bool Relocalization();
    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

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

    ORB_SLAM2::ORBextractor *mpORBextractor;
    ORB_SLAM2::ORBVocabulary *mpORBVocabulary;

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