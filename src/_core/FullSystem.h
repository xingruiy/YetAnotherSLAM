#ifndef _SLAM_SYSTEM_H
#define _SLAM_SYSTEM_H

#include <unistd.h>
#include <memory>
#include <thread>
#include <iostream>
#include <ORBVocabulary.h>

namespace slam
{

class Viewer;
class Tracking;
class Map;
class Frame;
class MapDrawer;
class KeyFrame;
class ORBextractor;
class LoopClosing;
class LocalMapping;
class KeyFrameDatabase;
struct FrameMetaData;

class FullSystem
{
public:
    ~FullSystem();
    FullSystem(const std::string &strSettings, const std::string &strVoc);
    void addImages(cv::Mat img, cv::Mat depth, double timestamp);
    void reset();
    void FuseAllMapStruct();
    void SaveTrajectoryTUM(const std::string &filename);
    void SaveKeyFrameTrajectoryTUM(const std::string &filename);

protected:
    void traceKeyFramePoints();
    void readSettings(const std::string &filename);

    // Map *mpMap;
    Tracking *mpTracker;
    Viewer *mpViewer;
    LocalMapping *localMapper;
    LoopClosing *loopCloser;
    MapDrawer *mpMapDrawer;

    KeyFrameDatabase *mpKeyFrameDB;
    ORBVocabulary *OrbVoc;
    ORBextractor *OrbExt;

    cv::Mat grayScale;
    cv::Mat depthFloat;

    std::vector<KeyFrame *> allKeyFramesHistory;
    std::vector<FrameMetaData *> allFrameHistory;
    std::vector<std::thread *> allChildThreads;

    Map *mpMap;

    Frame *newF;
};

} // namespace slam

#endif