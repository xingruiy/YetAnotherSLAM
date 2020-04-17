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
class KeyFrame;
class ORBextractor;
class LoopClosing;
class LocalMapping;
class BaseIOWrapper;
class KeyFrameDatabase;
struct FrameMetaData;

class FullSystem
{
public:
    ~FullSystem();
    FullSystem(const std::string &strSettings, const std::string &strVoc);
    void reset();
    void shutdown();
    void addOutput(BaseIOWrapper *io);

    void addImages(cv::Mat img, cv::Mat depth, double timestamp);

    void FuseAllMapStruct();
    void SaveTrajectoryTUM(const std::string &filename);
    void SaveKeyFrameTrajectoryTUM(const std::string &filename);

protected:
    void traceKeyFramePoints();
    void readSettings(const std::string &filename);

    Map *mpMap;
    Tracking *mpTracker;
    LocalMapping *localMapper;
    LoopClosing *loopCloser;

    KeyFrameDatabase *mpKeyFrameDB;
    ORBVocabulary *OrbVoc;
    ORBextractor *OrbExt;

    std::vector<BaseIOWrapper *> outputs;
    std::vector<KeyFrame *> allKeyFramesHistory;
    std::vector<FrameMetaData *> allFrameHistory;
    std::vector<std::thread *> allChildThreads;
};

} // namespace slam

#endif