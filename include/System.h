#pragma once
#include <memory>
#include <thread>
#include <iostream>

#include "Map.h"
#include "Frame.h"
#include "Viewer.h"
#include "KeyFrame.h"
#include "Tracking.h"
#include "Mapping.h"
#include "GlobalDef.h"
#include "LoopFinder.h"

namespace SLAM
{

class Viewer;
class Mapping;
class Tracking;
class MapViewer;
class LoopFinder;

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

    std::thread *mappingThread;
    std::thread *viewerThread;
    std::thread *loopThread;

    Map *mpMap;
    Tracking *tracker;
    Viewer *viewer;
    Mapping *mapping;
    LoopFinder *loopClosing;

    cv::Mat grayScale;
    cv::Mat depthFloat;
};

} // namespace SLAM