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
    void trackNewFrame(cv::Mat img, cv::Mat depth, double ts);
    void reset();

    void SetLocalMapper(LocalMapping *pLocalMapper);
    void SetLoopClosing(LoopClosing *pLoopClosing);
    void SetViewer(Viewer *pViewer);

public:
    enum eTrackingState
    {
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };

    eTrackingState mState;

    Frame currFrame;

    // Lists used to recover the full camera trajectory
    std::list<Sophus::SE3d> mlRelativeFramePoses;
    std::list<KeyFrame *> mlpReferences;
    std::list<double> mlFrameTimes;
    std::list<bool> mlbLost;

protected:
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
    LocalMapping *localMapper;
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
    ORBextractor *ORBExt;
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