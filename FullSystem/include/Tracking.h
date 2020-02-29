#pragma once
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "Viewer.h"
#include "KeyFrame.h"
#include "LocalMapping.h"
#include "System.h"
#include "GlobalDef.h"
#include "LoopClosing.h"
#include "MeshEngine.h"
#include "RGBDTracking.h"

namespace SLAM
{

class Viewer;
class System;
class KeyFrame;
class LocalMapping;

class Tracking
{
public:
    Tracking(System *pSystem, Map *pMap);

    // Preprocess the input and call Track().
    void GrabImageRGBD(cv::Mat ImGray, cv::Mat Depth, const double TimeStamp);

    void SetLocalMapper(LocalMapping *pLocalMapper);
    void SetLoopClosing(LoopClosing *pLoopClosing);
    void SetViewer(Viewer *pViewer);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

public:
    // Tracking states
    enum eTrackingState
    {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    std::list<Eigen::Matrix4d> mlRelativeFramePoses;
    std::list<KeyFrame *> mlpReferences;
    std::list<double> mlFrameTimes;
    std::list<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void reset();

protected:
    // Main tracking function.
    void Track();

    // Map initialization
    void InitializeSystem();

    bool Relocalization();
    bool TrackRGBD();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    //Other Thread Pointers
    LocalMapping *mpLocalMapper;
    LoopClosing *mpLoopClosing;

    // Dense Tracking And Mapping
    MeshEngine *mpMeshEngine;
    RGBDTracking *mpTracker;

    // System
    System *mpSystem;

    //Drawers
    Viewer *mpViewer;
    Sophus::SE3d mReferenceFramePose;

    //Map
    Map *mpMap;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame *mpCurrentKeyFrame;
    KeyFrame *mpLastKeyFrame;
    Frame mLastFrame;
    ORBextractor *mpExtractor;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    // Voxel Map Structure
    MapStruct *mpCurrentMapStruct;
};

} // namespace SLAM