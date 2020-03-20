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
#include "MapDrawer.h"
#include "RGBDTracking.h"
#include "RayTraceEngine.h"
#include "KeyFrameDatabase.h"

class LocalBundler;

namespace SLAM
{

class Viewer;
class System;
class KeyFrame;
class LocalMapping;

class Tracking
{
public:
    Tracking(System *pSystem, ORBVocabulary *pVoc, Map *pMap, KeyFrameDatabase *pKFDB);

    // Preprocess the input and call Track().
    void GrabImageRGBD(cv::Mat ImGray, cv::Mat Depth, const double TimeStamp);

    void SetLocalMapper(LocalMapping *pLocalMapper);
    void SetLoopClosing(LoopClosing *pLoopClosing);
    void SetViewer(Viewer *pViewer);
    void SetMapDrawer(MapDrawer *pMapDrawer);

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
    void StereoInitialization();

    bool Relocalization();
    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool TrackRGBD();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // Other Thread Pointers
    LocalMapping *mpLocalMapper;
    LoopClosing *mpLoopClosing;

    // System
    System *mpSystem;

    //BoW
    ORBVocabulary *mpORBVocabulary;
    KeyFrameDatabase *mpKeyFrameDB;

    //Local Map
    KeyFrame *mpReferenceKF;
    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;

    //Drawers
    Viewer *mpViewer;
    Sophus::SE3d mReferenceFramePose;

    //Map
    Map *mpMap;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame *mpCurrentKeyFrame;
    KeyFrame *mpLastKeyFrame;
    Frame mLastFrame;
    ORBextractor *mpORBExtractor;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    // The following sections are added
public:
    // Dense Tracking And Mapping
    RayTraceEngine *mpRayTraceEngine;
    MeshEngine *mpMeshEngine;
    RGBDTracking *mpTracker;

    // Voxel Map Structure
    MapStruct *mpCurrentMapStruct;

    // Local bundler
    LocalBundler *bundler;

    // Raw depth for fusion
    cv::cuda::GpuMat mRawDepth;

    std::vector<KeyFrame *> mvKeyframeHist;
    MapDrawer *mpMapDrawer;
};

} // namespace SLAM