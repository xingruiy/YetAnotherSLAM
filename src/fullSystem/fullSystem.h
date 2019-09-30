#pragma once
#include <memory>
#include <thread>
#include <iostream>
#include "utils/map.h"
#include "utils/numType.h"
#include "loopClosure/loopCloser.h"
#include "localMapper/localMapper.h"
#include "optimizer/localOptimizer.h"
#include "denseTracker/denseTracker.h"
#include "mapViewer/mapViewer.h"

class FullSystem
{
    const bool viewerEnabled;
    const size_t maxNumRelocAttempt = 3;

    bool needNewKF();
    void createNewKF();
    bool trackCurrentFrame();
    void fuseCurrentFrame();
    void raytraceCurrentFrame();
    bool tryRelocalizeCurrentFrame(bool updatePoints);

    std::shared_ptr<Map> map;
    std::shared_ptr<LoopCloser> loopCloser;
    std::shared_ptr<DenseMapping> localMapper;
    std::shared_ptr<DenseTracker> coarseTracker;
    std::shared_ptr<LocalOptimizer> localOptimizer;

    std::shared_ptr<Frame> currentFrame;
    std::shared_ptr<Frame> currentKeyframe;

    std::vector<SE3> rawFramePoseHistory;
    std::vector<SE3> rawKeyFramePoseHistory;

    SE3 lastTrackedPose;
    SE3 accumulateTransform;

    GMat bufferFloatwxh;
    GMat bufferVec4wxh;

    std::thread loopThread;
    std::thread localOptThread;
    MapViewer *viewer;

    int currentState;

public:
    ~FullSystem();
    FullSystem(const char *configFile);
    FullSystem(
        int w, int h,
        Mat33d K,
        int numLvl,
        bool enableViewer = true);
    void resetSystem();
    void processFrame(Mat rawImage, Mat rawDepth);

    std::vector<SE3> getFramePoseHistory();
    std::vector<SE3> getKeyFramePoseHistory();
    std::vector<SE3> getRawFramePoseHistory() const;
    std::vector<SE3> getRawKeyFramePoseHistory() const;
    std::vector<Vec3f> getMapPointPosAll();

    void setMapViewerPtr(MapViewer *viewer);
    size_t getMesh(float *vbuffer, float *nbuffer, size_t bufferSize);
};