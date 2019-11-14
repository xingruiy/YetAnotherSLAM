#pragma once
#include <set>
#include <mutex>
#include <memory>
#include <vector>
#include "utils/numType.h"
#include "dataStruct/frame.h"
#include "dataStruct/keyFrame.h"
#include "dataStruct/mapPoint.h"

class Map
{
public:
    Map();

    void clear();

    // TODO: not yet implemented
    void writeToDisk(const char *fileName);

    // TODO: not yet implemented
    void readFromDisk(const char *fileName);

    // Insert new key frame into the map
    void addKeyFrame(std::shared_ptr<KeyFrame> KF);

    // Insert new map point into the map
    void addMapPoint(std::shared_ptr<MapPoint> MP);

    // Get map ponit pos
    void getMapPoint(std::vector<Vec3f> &MPs);

    // void addFramePoseRaw(const SE3 &T);
    // void addKeyframePoseRaw(const SE3 &T);
    // void addFramePose(const SE3 &T, std::shared_ptr<Frame> keyFrame);

    // std::shared_ptr<Frame> getCurrentKeyframe() const;
    // void setCurrentKeyframe(std::shared_ptr<Frame> keyFrame);
    // void addUnprocessedKeyframe(std::shared_ptr<Frame> keyFrame);
    // std::shared_ptr<Frame> getUnprocessedKeyframe();
    // void addLoopClosingKeyframe(std::shared_ptr<Frame> keyFrame);
    // std::shared_ptr<Frame> getLoopClosingKeyframe();

    // std::vector<SE3> getKeyframePoseRaw();
    // std::vector<SE3> getKeyframePoseOptimized();
    // std::vector<SE3> getFramePoseRaw();
    // std::vector<SE3> getFramePoseOptimized();
    // std::vector<Vec3f> getMapPointVec3All();

public:
    Mat descriptorDB;
    shared_vector<MapPoint> mapPointDB;
    shared_vector<KeyFrame> keyFrameDB;

    std::mutex mapMutex;
    std::shared_ptr<Frame> currentKeyframe;

    // std::mutex mutexkf, mutexloop;
    // std::queue<std::shared_ptr<Frame>> unprocessedKeyframeQueue;
    // std::queue<std::shared_ptr<Frame>> loopClosingKeyframeQueue;

    // std::mutex histMutex;
    // std::vector<SE3> keyframePoseRawAll;
    // std::vector<SE3> framePoseRawAll;
    // std::vector<std::pair<SE3, std::shared_ptr<Frame>>> framePoseOptimized;
};