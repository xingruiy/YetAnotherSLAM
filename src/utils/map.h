#pragma once
#include <set>
#include <mutex>
#include <memory>
#include <vector>
#include "utils/numType.h"
#include "utils/frame.h"
#include "utils/mapPoint.h"

class Map
{
    std::mutex mapMutex;
    Mat pointDescriptorsAll;
    std::shared_ptr<Frame> currentKeyframe;
    std::vector<std::shared_ptr<Frame>> keyframesAll;
    std::vector<std::shared_ptr<MapPoint>> mapPointsAll;

    std::mutex mutexkf, mutexloop;
    std::queue<std::shared_ptr<Frame>> unprocessedKeyframeQueue;
    std::queue<std::shared_ptr<Frame>> loopClosingKeyframeQueue;

    std::mutex histMutex;
    std::vector<SE3> keyframePoseRawAll;
    std::vector<SE3> framePoseRawAll;
    std::vector<std::pair<SE3, std::shared_ptr<Frame>>> framePoseOptimized;

public:
    Map();
    void clear();
    void addKeyFrame(std::shared_ptr<Frame> kf);
    Mat getPointDescriptorsAll() const;
    std::shared_ptr<Frame> getCurrentKeyframe() const;
    void setCurrentKeyframe(std::shared_ptr<Frame> kf);
    void addMapPoint(std::shared_ptr<MapPoint> pt);

    void addFramePoseRaw(const SE3 &T);
    void addKeyframePoseRaw(const SE3 &T);
    void addFramePose(const SE3 &T, std::shared_ptr<Frame> kf);

    void addUnprocessedKeyframe(std::shared_ptr<Frame> kf);
    std::shared_ptr<Frame> getUnprocessedKeyframe();
    void addLoopClosingKeyframe(std::shared_ptr<Frame> kf);
    std::shared_ptr<Frame> getLoopClosingKeyframe();

    std::vector<SE3> getKeyframePoseRaw();
    std::vector<SE3> getKeyframePoseOptimized();
    std::vector<SE3> getFramePoseRaw();
    std::vector<SE3> getFramePoseOptimized();
    std::vector<Vec3f> getMapPointVec3All();

    const std::vector<std::shared_ptr<Frame>> &getKeyframesAll();
    const std::vector<std::shared_ptr<MapPoint>> &getMapPointsAll();
    std::vector<std::shared_ptr<Frame>> getLastNKeyframes(const size_t N);
};