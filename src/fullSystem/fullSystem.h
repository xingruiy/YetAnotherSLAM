#pragma once
#include <memory>
#include <thread>
#include <iostream>
#include "utils/numType.h"
#include "dataStruct/map.h"
#include "dataStruct/frame.h"
#include "dataStruct/keyFrame.h"
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
    FullSystem(int w, int h, Mat33d K, int numLvl, MapViewer &viewer);

    // Main entry point
    void processFrame(Mat imRGB, Mat imDepth);

    // Reset the system to its initial state
    void resetSystem();

    // toggle mapping
    void setMappingEnable(const bool enable);

    // Set system to relocalisation mode
    // TODO: can't resume from lost yet
    void setSystemStateToLost();

    // Set system to debugging mode
    void setSystemStateToTest();
    void setGraphMatching(const bool &flag);
    void setGraphGetNormal(const bool &flag);
    void testNextKF();

    // Reuse some of the data
    void setCpuBufferVec4FloatWxH(Mat buffer);

    // For visualization
    size_t getMesh(float *vbuffer, float *nbuffer, size_t bufferSize);
    std::vector<SE3> getFramePoseHistory();
    std::vector<SE3> getKeyFramePoseHistory();
    std::vector<Vec3f> getMapPointPosAll();

public:
    bool needNewKF();
    void createNewKF();
    bool trackCurrentFrame();
    void fuseCurrentFrame();
    void raytraceCurrentFrame();
    bool tryRelocalizeCurrentFrame();
    bool tryRelocalizeKeyframe(std::shared_ptr<Frame> kf);

    // System components
    std::thread loopClosureThread;
    std::thread localMappingThread;
    std::shared_ptr<Map> map;
    std::shared_ptr<DenseMapping> denseMapper;
    std::shared_ptr<DenseTracker> coarseTracker;
    std::shared_ptr<LocalMapper> localMapper;

    MapViewer *viewer;

    std::shared_ptr<Frame> currentFrame;
    std::shared_ptr<Frame> currentKeyframe;

    // System parameters
    int imageWidth;
    int imageHeight;
    Mat33d K;

    // Reusable buffers
    GMat gpuBufferFloatWxH;
    GMat gpuBufferVec4FloatWxH;
    GMat gpuBufferVec4FloatWxH2;
    Mat cpuBufferVec4FloatWxH;
    Mat cpuBufferFloatWxH;
    Mat cpuBufferVec3ByteWxH;
    Mat cpuBufferVec3FloatWxH;

    // System statistics
    void resetStatistics();
    size_t numFramesProcessed;
    size_t numRelocalisationAttempted;

    // Accumulated States
    // these needs to be cleared when reset
    SE3 rawTransformation;
    SE3 lastTrackedPose;

    // State machine
    SystemState lastState, state;
    std::shared_ptr<KeyFrame> currKeyFrame;
    std::shared_ptr<KeyFrame> lastKeyFrame;

    // **For debugging relocalization
    // will be removed later
    size_t testKFId;
    size_t lastTestedKFId;
    bool mappingEnabled;
    bool useGraphMatching;
    bool shouldCalculateNormal;
};