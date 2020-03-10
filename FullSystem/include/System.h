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

namespace SLAM
{

class Viewer;
class Tracking;
class MapViewer;
class LoopClosing;
class LocalMapping;

class System
{
public:
    ~System();
    System(const std::string &strSettingFile, const std::string &strVocFile);
    void TrackRGBD(cv::Mat img, cv::Mat depth, const double timeStamp);
    void Shutdown();
    void reset();
    void FuseAllMapStruct();

    void WriteToFile(const std::string &strFile);
    void ReadFromFile(const std::string &strFile);

private:
    void readSettings(const std::string &strSettingFile);
    void loadORBVocabulary(const std::string &strVocFile);

    std::thread *mpLocalMappingThread;
    std::thread *mpViewerThread;
    std::thread *mpLoopThread;

    Map *mpMap;
    Tracking *mpTracker;
    Viewer *mpViewer;
    LocalMapping *mpLocalMapper;
    LoopClosing *mpLoopClosing;
    MapDrawer *mpMapDrawer;

    KeyFrameDatabase *mpKeyFrameDB;
    ORBVocabulary *mpORBVocabulary;

    cv::Mat grayScale;
    cv::Mat depthFloat;
};

} // namespace SLAM