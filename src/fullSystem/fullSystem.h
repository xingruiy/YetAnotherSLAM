#pragma once
#include <memory>
#include <thread>
#include <iostream>
#include "utils/map.h"
#include "utils/numType.h"
#include "localMapper/localMapper.h"
#include "optimizer/localOptimizer.h"
#include "denseTracker/denseTracker.h"
#include "mapViewer/mapViewer.h"

enum class SystemState
{
    NotInitialized,
    OK,
    Lost
};

class FullSystem
{
    const bool viewerEnabled;
    const size_t maxNumRelocAttempt = 3;

    bool needNewKF();
    void createNewKF();
    bool trackCurrentFrame();
    void fuseCurrentFrame();
    void raytraceCurrentFrame();
    bool tryRelocalizeCurrentFrame();

    std::thread loopThread;
    std::thread localOptThread;
    MapViewer *viewer;

    std::shared_ptr<Map> map;
    std::shared_ptr<DenseMapping> localMapper;
    std::shared_ptr<DenseTracker> coarseTracker;
    std::shared_ptr<LocalOptimizer> localOptimizer;

    std::shared_ptr<Frame> currentFrame;
    std::shared_ptr<Frame> currentKeyframe;

    SE3 lastTrackedPose;
    SE3 accumulateTransform;

    GMat bufferFloatwxh;
    GMat bufferVec4wxh;
    GMat bufferVec4wxh2;
    Mat cbufferFloatwxh;
    Mat cbufferFloatVec3wxh;

    SystemState state;
    SystemState lastState;

    int imageWidth;
    int imageHeight;
    Mat33d camIntrinsics;
    bool mappingEnabled;
    bool useGraphMatching;
    bool shouldCalculateNormal;
    size_t numProcessedFrames;

public:
    ~FullSystem();
    FullSystem(
        int w, int h,
        Mat33d K,
        int numLvl,
        bool enableViewer = true);

    // allow viewing the map
    void setMapViewerPtr(MapViewer *viewer);

    // toggle mapping
    void setMappingEnable(const bool enable);

    // trigger relocalization
    // TODO: this is currently in infinite mode
    void setSystemStateToLost();
    void setGraphMatching(const bool &flag);
    void setGraphMatchingMethod(const bool &flag);

    // reset the system to its initial state
    void resetSystem();

    // main process function
    void processFrame(Mat rawImage, Mat rawDepth);
    size_t getMesh(float *vbuffer, float *nbuffer, size_t bufferSize);

    std::vector<SE3> getFramePoseHistory();
    std::vector<SE3> getKeyFramePoseHistory();
    std::vector<Vec3f> getMapPointPosAll();
};