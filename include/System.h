#pragma once
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
class LocalMapping;
class Tracking;
class MapViewer;
class LoopClosing;

class System
{
public:
    ~System();
    System(const std::string &strSettingFile, const std::string &strVocFile);
    void trackImage(cv::Mat img, cv::Mat depth, const double timeStamp);
    void kill();
    void reset();

private:
    void readSettings(const std::string &strSettingFile);
    void loadORBVocabulary(const std::string &strVocFile);

    std::thread *mpLocalMappingThread;
    std::thread *mpViewerThread;
    std::thread *mpLoopThread;

    Map *mpMap;
    Tracking *mpTracker;
    Viewer *mpViewer;
    LocalMapping *mpLocalMapping;
    LoopClosing *mpLoopClosing;
    MapDrawer *mpMapDrawer;

    KeyFrameDatabase *mpKeyFrameDB;
    ORB_SLAM2::ORBVocabulary *mpORBVocabulary;

    cv::Mat grayScale;
    cv::Mat depthFloat;
};

} // namespace SLAM