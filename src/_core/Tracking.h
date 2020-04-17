#pragma once
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "Viewer.h"
#include "KeyFrame.h"
#include "LocalMapping.h"
#include "FullSystem.h"
#include "GlobalSettings.h"
#include "LoopClosing.h"
#include "MeshEngine.h"
#include "CoarseTracking.h"
#include "RayTracer.h"
#include "KeyFrameDatabase.h"

namespace slam
{

class Viewer;
class FullSystem;
class KeyFrame;
class Map;
class LocalMapping;

class Tracking
{
public:
    Tracking(FullSystem *pSystem, ORBVocabulary *pVoc, Map *mpMap, KeyFrameDatabase *pKFDB);
    void trackNewFrame(Frame &F);
    void reset();

    void SetLocalMapper(LocalMapping *pLocalMapper);
    void SetLoopClosing(LoopClosing *pLoopClosing);
    void setIOWrapper(Viewer *pViewer);

public:
    bool hasLost = false;
    bool isInitialized = false;
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
    void setLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool addImages();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();
    void CreateNewMapPoints();

    // Other Thread Pointers
    LocalMapping *localMapper;
    LoopClosing *loopCloser;

    // FullSystem
    FullSystem *mpSystem;

    //BoW
    ORBVocabulary *OrbVoc;
    KeyFrameDatabase *mpKeyFrameDB;

    //Local Map
    KeyFrame *mpReferenceKF;
    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> localPoints;

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