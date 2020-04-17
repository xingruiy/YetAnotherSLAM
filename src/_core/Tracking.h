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
#include "CoarseTracking.h"
#include "RayTracer.h"
#include "KeyFrameDatabase.h"

namespace slam
{

class Viewer;
class System;
class KeyFrame;
class Map;
class LocalMapping;

class Tracking
{
public:
    Tracking(System *pSystem, ORBVocabulary *pVoc, Map *mpMap, KeyFrameDatabase *pKFDB);

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
    Frame currFrame;
    cv::Mat mImGray;

    // Lists used to recover the full camera trajectory
    std::list<Sophus::SE3d> mlRelativeFramePoses;
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
    void initSystem();

    bool Relocalization();
    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool takeNewFrame();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();
    void CreateNewMapPoints();

    // Other Thread Pointers
    LocalMapping *mpLocalMapper;
    LoopClosing *mpLoopClosing;

    // System
    System *mpSystem;

    //BoW
    ORBVocabulary *ORBVoc;
    KeyFrameDatabase *mpKeyFrameDB;

    //Local Map
    KeyFrame *mpReferenceKF;
    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;

    //Drawers
    Viewer *mpViewer;

    //Map
    Map *mpMap;

    //Last Frame
    KeyFrame *mpLastKeyFrame;
    Frame lastFrame;
    ORBextractor *mpORBExtractor;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastSuccessRelocFrameId;

    // The following sections are added
public:
    bool needNewVoxelMap();
    void createNewVoxelMap();

    // Dense Tracking And Mapping
    RayTracer *rayTracer;
    MeshEngine *mpMeshEngine;
    CoarseTracking *mpTracker;

    // Voxel Map Structure
    MapStruct *mpCurrVoxelMap;

    // Raw depth for fusion
    cv::cuda::GpuMat mRawDepth;
};

} // namespace slam