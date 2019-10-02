#pragma once
#include <mutex>
#include <memory>
#include "utils/mapPoint.h"
#include "utils/numType.h"
#include "optimizer/featureMatcher.h"

class MapPoint;
class FeatureMatcher;

class Frame
{
    Mat rawDepth;
    Mat rawImage;
    Mat rawIntensity;
    bool keyframeFlag;

    // Raw pose update from the tracker, this stays unchanged.
    SE3 relativePose;
    // Raw world pose used for local map registeration. Only for KFs.
    SE3 rawKeyframePose;
    // stores result from the optimizer, highly volatile
    SE3 optimizedPose;

    SE3 framePose;
    std::mutex poseMutex;

    size_t kfId;
    static size_t nextKFId;

    int imgWidth;
    int imgHeight;
    Mat33d camIntrinsics;
    size_t numPointsCreated;
    size_t numPointsDetectd;
    std::shared_ptr<Frame> referenceKF;

public:
    Frame();
    Frame(int w,
          int h,
          Mat33d &K,
          Mat colourImage,
          Mat depthImage,
          Mat intensityImage);

    int getImageWidth() const;
    int getImageHeight() const;
    Mat33d getIntrinsics() const;

    Mat getDepth() const;
    Mat getImage() const;
    Mat getIntensity() const;

    void flagKeyFrame();
    bool isKeyframe() const;
    size_t getId() const;

    SE3 getTrackingResult() const;
    SE3 getPoseInGlobalMap() const;
    SE3 getPoseInLocalMap() const;

    void setReferenceKF(std::shared_ptr<Frame> kf);
    std::shared_ptr<Frame> getReferenceKF() const;

    bool hasMapPoint() const;
    std::shared_ptr<MapPoint> createMapPoint(size_t idx);
    void setMapPoint(std::shared_ptr<MapPoint> pt, size_t idx);
    void eraseMapPoint(size_t idx);
    void detectKeyPoints(std::shared_ptr<FeatureMatcher> matcher);
    const std::vector<std::shared_ptr<MapPoint>> &getMapPoints() const;

    double *getParameterBlock();
    void setTrackingResult(const SE3 &T);
    void setRawKeyframePose(const SE3 &T);
    void setOptimizationResult(const SE3 &T);

    // TODO : refactory this
    Mat pointDesc;
    int kfIdLocalRoot;
    std::vector<float> depthVec;
    std::vector<cv::KeyPoint> cvKeyPoints;
    std::vector<std::shared_ptr<MapPoint>> mapPoints;
};