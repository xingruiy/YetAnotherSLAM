#pragma once
#include <memory>
#include <thread>
#include <iostream>
#include "dataStruct/map.h"
#include "utils/numType.h"
#include "mapViewer/mapViewer.h"
#include "denseMapper/denseMapper.h"
#include "localMapper/localMapper.h"
#include "denseTracker/denseTracker.h"

enum class SystemState
{
    NotInitialized,
    OK,
    Lost,
    Test
};

class FullSystem
{
public:
    ~FullSystem();
    FullSystem(int w, int h, Mat33d K, int numLvl, bool enableViewer = true);
    // allow viewing the map
    void setMapViewerPtr(MapViewer *viewer);
    // toggle mapping
    void setMappingEnable(const bool enable);
    // trigger relocalization
    // TODO: can't resume from lost
    void setSystemStateToLost();
    void setSystemStateToTest();
    // relocalization related parameters
    void setGraphMatching(const bool &flag);
    void setGraphGetNormal(const bool &flag);
    // reset the system to its initial state
    void resetSystem();
    void testNextKF();
    // main process function
    void setCurrentNormal(GMat nmap);
    void processFrame(Mat imRGB, Mat imDepth);

    std::vector<SE3> getFramePoseHistory();
    std::vector<SE3> getKeyFramePoseHistory();
    std::vector<Vec3f> getMapPointPosAll();

    size_t getMesh(
        float *vbuffer,
        float *nbuffer,
        size_t bufferSize);

public:
    const bool viewerEnabled;
    int numTimesRun;

    bool needNewKF();
    void createNewKF();
    bool trackCurrentFrame();
    void fuseCurrentFrame();
    void raytraceCurrentFrame();
    bool tryRelocalizeCurrentFrame();
    bool tryRelocalizeKeyframe(std::shared_ptr<Frame> kf);

    std::thread loopThread;
    std::thread localOptThread;
    MapViewer *viewer;

    std::shared_ptr<Map> map;
    std::shared_ptr<DenseMapping> denseMapper;
    std::shared_ptr<DenseTracker> coarseTracker;
    std::shared_ptr<FeatureMapper> localMapper;

    std::shared_ptr<Frame> currentFrame;
    std::shared_ptr<Frame> currentKeyframe;

    SE3 lastTrackedPose;
    SE3 accumulateTransform;

    // State machine
    SystemState state, lastState;

    int imageWidth;
    int imageHeight;
    Mat33d camIntrinsics;
    bool mappingEnabled;
    bool useGraphMatching;
    bool shouldCalculateNormal;

    // **For debugging relocalization
    size_t testKFId;
    size_t lastTestedKFId;

    // Reusable buffers
    GMat gpuBufferFloatWxH;
    GMat gpuBufferVec4FloatWxH;
    GMat gpuBufferVec4FloatWxH2;
    Mat cpuBufferVec4FloatWxH;
    Mat cpuBufferFloatWxH;
    Mat cpuBufferVec3ByteWxH;
    Mat cpuBufferVec3FloatWxH;

    // System statistics
    size_t numProcessedFrames;
};