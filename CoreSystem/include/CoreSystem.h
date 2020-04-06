#pragma once
#include <unistd.h>
#include <memory>
#include <thread>
#include <iostream>
#include <ORBVocabulary.h>

#include "Map.h"
#include "Frame.h"
#include "Viewer.h"
#include "KeyFrame.h"
#include "Tracking.h"
#include "LocalMapping.h"
#include "GlobalDef.h"
#include "LoopClosing.h"
#include "MapDrawer.h"
#include "KeyFrameDatabase.h"

namespace slam
{

class Viewer;
class Tracking;
class MapViewer;
class MapManager;
class LoopClosing;
class LocalMapping;
class GlobalSettings;

class CoreSystem
{
public:
    CoreSystem(GlobalSettings *settings, const std::string &strVocFile);
    ~CoreSystem();

    void takeNewFrame(cv::Mat img, cv::Mat depth, const double timeStamp);

    void Shutdown();
    void reset();
    void FuseAllMapStruct();
    void DisplayNextMap();

    void writeTrajectoryToFile(const std::string &filename);

private:
    void readSettings(const std::string &strSettingFile);

    std::thread *mpLocalMappingThread;
    std::thread *mpViewerThread;
    std::thread *mpLoopThread;

    // Map *mpMap;
    Tracking *localTracker;
    Viewer *mpViewer;
    LocalMapping *localMapper;
    LoopClosing *loopCloser;
    MapDrawer *mpMapDrawer;

    KeyFrameDatabase *KFDatabase;
    ORBVocabulary *ORBVoc;

    cv::Mat grayScale;
    cv::Mat depthFloat;

    MapManager *mapManager;
    GlobalSettings *settings;
};

} // namespace slam