#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "DENSE/include/DenseTracking.h"
#include "DENSE/include/DenseMapping.h"

#include "Frame.h"
#include "Viewer.h"
#include "LocalMapping.h"
#include "FullSystem.h"

class Viewer;
class LocalMapping;
class FullSystem;

class Tracking
{
public:
    Tracking(const std::string &strSettingsFile, FullSystem *pSys, Map *pMap, ORB_SLAM2::ORBVocabulary *pVoc);

    void TrackImageRGBD(const cv::Mat &imGray, const cv::Mat &imDepth);

    void SetViewer(Viewer *pViewer);
    void SetLocalMapper(LocalMapping *pLocalMapper);

    void Reset();

private:
    enum class TrackingState
    {
        NOTInit,
        OK,
        LOST
    };

    void InitializeTracking();
    bool TrackLastFrame();
    bool Relocalization();
    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();
    // int CheckObservations();

    // Local map management
    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    // Calibration
    Eigen::Matrix3d mK;
    int mImgWidth;
    int mImgHeight;
    float mbf;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    TrackingState meState;
    TrackingState meLastState;

    // Dense Tracker
    DenseTracking *mpTracker;

    // Sparse Mapping
    LocalMapping *mpLocalMapper;

    // Dense Mapping
    DenseMapping *mpMapper;

    // Disable mapping
    bool mbOnlyTracking;

    double mDepthScaleFactor;

    // Frames
    Frame mCurrentFrame;
    Frame mLastFrame;

    // ORB
    ORB_SLAM2::ORBextractor *mpORBextractor;

    // BoW
    ORB_SLAM2::ORBVocabulary *mpORBVocabulary;

    // Used for local map
    KeyFrame *mpReferenceKF;
    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;

    // System
    FullSystem *mpFullSystem;

    // Map
    Map *mpMap;

    // Map Viewr
    Viewer *mpViewer;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;
};