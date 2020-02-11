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
    Tracking(const std::string &strSettingsFile, System *pSys, Map *pMap);
    void TrackImage(const cv::Mat &imGray, const cv::Mat &imDepth, const double &TimeStamp);

    void SetViewer(Viewer *pViewer);
    void SetLocalMapper(Mapping *pLocalMapper);

    void Reset();

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

    // Calibration
    cv::Mat mDistCoef;
    Eigen::Matrix3d mK;
    int mImgWidth;
    int mImgHeight;
    int mMaxFrameRate;
    float mbf;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    TrackingState mTrackingState;

    // Dense Tracker
    DenseTracking *mpTracker;

    // Sparse Mapping
    Mapping *mpMapping;

    // Dense Mapping
    DenseMapping *mpMapper;

    // Disable mapping
    bool mbOnlyTracking;

    double mDepthScaleFactor;

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
    Viewer *mpViewer;

    // Used for local map
    KeyFrame *mpReferenceKF;
    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;
};

} // namespace SLAM