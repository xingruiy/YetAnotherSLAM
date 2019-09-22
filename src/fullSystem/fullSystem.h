#pragma once
#include <memory>
#include <thread>
#include <iostream>
#include "utils/numType.h"
#include "localMapper/localMapper.h"
#include "globalMapper/globalMapper.h"
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

    std::shared_ptr<GlobalMapper> globalMapper;
    std::shared_ptr<DenseMapping> localMapper;
    std::shared_ptr<DenseTracker> coarseTracker;

    std::shared_ptr<Frame> currentFrame;
    std::shared_ptr<Frame> referenceFrame;

    std::vector<SE3> rawFramePoseHistory;
    std::vector<SE3> rawKeyFramePoseHistory;

    SE3 lastTrackedPose;
    SE3 lastReferencePose;
    SE3 accumulateTransform;

    GMat bufferFloatwxh;
    GMat bufferVec4wxh;

    std::thread optThread;

public:
    ~FullSystem();
    FullSystem(const char *configFile);
    FullSystem(int w, int h, Mat33d K, int numLvl, bool view);
    void resetSystem();
    void processFrame(Mat rawImage, Mat rawDepth);

    std::vector<Vec3f> getActiveKeyPoints();
    std::vector<Vec3f> getStableKeyPoints();
    std::vector<SE3> getFramePoseHistory();
    std::vector<SE3> getKeyFramePoseHistory();
    std::vector<SE3> getRawFramePoseHistory() const;
    std::vector<SE3> getRawKeyFramePoseHistory() const;
    size_t getMesh(float *vbuffer, float *nbuffer, size_t bufferSize);
};