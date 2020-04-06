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

class CoreSystem
{
public:
    ~CoreSystem();
    CoreSystem(const std::string &strSettingFile, const std::string &strVocFile);
    void takeNewFrame(cv::Mat img, cv::Mat depth, const double timeStamp);
    void Shutdown();
    void reset();
    void FuseAllMapStruct();
    void DisplayNextMap();

    void WriteToFile(const std::string &strFile);
    void ReadFromFile(const std::string &strFile);
    void writeTrajectoryToFile(const std::string &filename);

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
    ORBVocabulary *ORBVoc;

    cv::Mat grayScale;
    cv::Mat depthFloat;

    MapManager *mpMapManager;
};

} // namespace slam