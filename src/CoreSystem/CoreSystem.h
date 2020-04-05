#ifndef CORE_SYSTEM_H
#define CORE_SYSTEM_H

#include <unistd.h>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "ORBVocabulary.h"

namespace slam
{

class Viewer;
class MapDrawer;
class Tracking;
class MapViewer;
class MapManager;
class LoopClosing;
class LocalMapping;
class GlobalSettings;
class BaseOutput;
class MapPoint;
class Frame;
class KeyFrame;
class CoarseTracking;
class KeyFrameDatabase;

class CoreSystem
{
public:
    ~CoreSystem();
    CoreSystem(const std::string &strSettingFile, const std::string &strVocFile);
    void TrackRGBD(cv::Mat img, cv::Mat depth, const double timeStamp);
    void Shutdown();
    void reset();
    void FuseAllMapStruct();
    void DisplayNextMap();

    void WriteToFile(const std::string &strFile);
    void ReadFromFile(const std::string &strFile);

private:
    void readSettings(const std::string &strSettingFile);
    void loadORBVocabulary(const std::string &strVocFile);

    std::thread *mpLocalMappingThread;
    std::thread *mpViewerThread;
    std::thread *mpLoopThread;

    // Map *mpMap;
    Tracking *mpTracker;
    Viewer *mpViewer;
    LocalMapping *mpLocalMapper;
    LoopClosing *mpLoopClosing;
    MapDrawer *mpMapDrawer;

    KeyFrameDatabase *mpKeyFrameDB;
    ORBVocabulary *mpORBVocabulary;

    cv::Mat grayScale;
    cv::Mat depthFloat;

    MapManager *mpMapManager;

public:
    CoreSystem(GlobalSettings *settings, const std::string &strVocFile);
    void takeNewImages(float *img, float *depth, const double ts);

    void blockUntilReset();
    void blockUntilFinished();

    inline void engageShutdown() { emergencyBreak = true; }
    inline void takeNewIOSystem(BaseOutput *baseIO) { outputs.push_back(baseIO); }

protected:
    void makeKeyFrame();

    // ======== multi threading ========
    void loopClosingThread();
    void localMappingThread();
    bool tellChildThreadsToWrapup = false;
    std::vector<std::thread *> childThreads;

    // ======== output wrappers ========
    std::vector<BaseOutput *> outputs;

    // ======== log files (optional) ========
    std::ofstream *rawLogs;

    // ======== calibrations ========
    int width, height;
    float fx, fy, cx, cy, ifx, ify;

    // ======== maps (protected by map mutex) ========
    MapManager *maps;
    std::vector<Frame *> allFrames;
    KeyFrameDatabase *KFDatabase;

    // ======== coarse tracking ========
    GlobalSettings *settings;
    CoarseTracking *coarseTracker;
    CoarseTracking *coarseTracker_bak;

    ORBVocabulary *ORBVocab;

    bool hasLost = false;
    bool needKF = false;
    bool hasInitialized = false;
    bool emergencyBreak = false;
};

} // namespace slam

#endif