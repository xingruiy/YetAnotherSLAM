#pragma once
#include <mutex>
#include <memory>
#include "utils/numType.h"
#include "dataStruct/mapPoint.h"
#include "localMapper/featureMatcher.h"

class MapPoint;
class FeatureMatcher;

class Frame
{
public:
    Frame();
    Frame(Mat imRGB, Mat imDepth, Mat imGray, Mat33d &K);

    Mat33d getIntrinsics() const;

    Mat getDepth() const;
    Mat getImage() const;
    Mat getOGDepth() const;
    Mat getIntensity() const;

    void flagKeyFrame();
    bool isKeyframe() const;
    size_t getId() const;

    SE3 getTrackingResult() const;
    SE3 getPoseInGlobalMap() const;
    SE3 getPoseInLocalMap() const;
    Vec3d getPositionWorld() const;
    void setReferenceKF(std::shared_ptr<Frame> kf);
    std::shared_ptr<Frame> getReferenceKF() const;

    // map related operations
    bool hasMapPoint() const;
    std::shared_ptr<MapPoint> createMapPoint(size_t idx);
    void setMapPoint(std::shared_ptr<MapPoint> pt, size_t idx);
    void eraseMapPoint(size_t idx);
    void detectKeyPoints(std::shared_ptr<FeatureMatcher> matcher);
    const std::vector<std::shared_ptr<MapPoint>> &getMapPoints() const;

    // used for bundle adjustment
    double *getParameterBlock();
    void setTrackingResult(const SE3 &T);
    void setRawKeyframePose(const SE3 &T);
    void setOptimizationResult(const SE3 &T);

    // TODO : refactory this
    Mat descriptors;
    std::vector<Vec3f> keyPointNorm;
    std::vector<float> keyPointDepth;
    std::vector<cv::KeyPoint> cvKeyPoints;
    std::vector<std::shared_ptr<MapPoint>> mapPoints;

    Mat getNormalMap() const;
    void setNormalMap(const Mat nmap);

public:
    Mat imDepth, imRGB;
    Mat ogDepth;
    Mat nmap;
    Mat rawIntensity;
    bool keyframeFlag;

    // Raw pose update from the tracker, this stays unchanged.
    SE3 relativePose;
    // Raw world pose used for local map registeration. Only for KFs.
    SE3 rawKeyframePose;
    // stores result from the loopCloser, highly volatile
    SE3 optimizedPose;

    SE3 framePose;
    std::mutex poseMutex;

    size_t kfId;
    static size_t nextKFId;

    Mat33d K;
    std::shared_ptr<Frame> referenceKF;
};
