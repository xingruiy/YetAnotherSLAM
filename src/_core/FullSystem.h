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
class BoWDatabase;
struct FrameMetaData;

class FullSystem
{
public:
    ~FullSystem();
    FullSystem(const std::string &strSettings, const std::string &strVoc);
    void reset();
    void shutdown();
    void initSystem(Frame *newF);
    void addOutput(BaseIOWrapper *io);

    void addImages(cv::Mat img, cv::Mat depth, double timestamp);
    void deliverTrackedFrame(Frame *newF, bool makeKF);

    void FuseAllMapStruct();
    void SaveTrajectoryTUM(const std::string &filename);
    void SaveKeyFrameTrajectoryTUM(const std::string &filename);

protected:
    void traceKeyFramePoints();
    void readSettings(const std::string &filename);

    bool hasLost = false;
    bool hasInitialized = false;
    void trackFrameCoarse(Frame *newF);

    void threadLoopClosing();
    void threadLocalMapping();

    // ==== global feature map, protected by map mutex
    Map *mpMap;

    // ==== main tracking thread, taking care of frame pose
    Tracking *mpTracker;

    // ==== local mapping for local BA
    LocalMapping *localMapper;

    // ==== loop closure and global BA
    LoopClosing *loopCloser;
    ORBVocabulary *OrbVoc;
    ORBextractor *OrbExt;
    BoWDatabase *mpKeyFrameDB;

    std::vector<BaseIOWrapper *> outputs;
    std::vector<KeyFrame *> allKeyFramesHistory;
    std::vector<FrameMetaData *> allFrameHistory;
    std::vector<std::thread *> allChildThreads;
};

} // namespace slam

#endif