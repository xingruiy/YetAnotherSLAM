#pragma once
#include <memory>
#include <iostream>
#include "utils/numType.h"
#include "localMapper/localMapper.h"
#include "denseTracker/denseTracker.h"

class FullSystem
{
    int currentState;
    bool enableMapViewer;

    bool needNewKF();
    void createNewKF();
    bool trackCurrentFrame();
    void fuseCurrentFrame(const SE3 &T);
    void updateLocalMapObservation(const SE3 &T);

    std::shared_ptr<DenseMapping> localMapper;
    std::shared_ptr<DenseTracker> coarseTracker;

    std::shared_ptr<Frame> currentFrame;
    std::shared_ptr<Frame> referenceFrame;

    std::vector<SE3> rawFramePoseHistory;

    SE3 lastTrackedPose;
    SE3 lastReferencePose;

public:
    FullSystem(const char *configFile);
    FullSystem(int w, int h, Mat33d K, int numLvl, bool view);
    void resetSystem();
    void processFrame(Mat rawImage, Mat rawDepth);

    size_t getMesh(float *vbuffer, float *nbuffer, size_t bufferSize);
    std::vector<SE3> getRawFramePoseHistory() const;
};