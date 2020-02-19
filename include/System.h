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
#include "Mapping.h"
#include "GlobalDef.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"

namespace SLAM
{

class Viewer;
class Mapping;
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

    std::thread *mappingThread;
    std::thread *viewerThread;
    std::thread *loopThread;

    Map *mpMap;
    Tracking *tracker;
    Viewer *viewer;
    Mapping *mapping;
    LoopClosing *loopClosing;

    KeyFrameDatabase *mpKeyFrameDB;
    ORB_SLAM2::ORBVocabulary *mpORBVocabulary;

    cv::Mat grayScale;
    cv::Mat depthFloat;
};

} // namespace SLAM