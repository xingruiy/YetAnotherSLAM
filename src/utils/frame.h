#pragma once
#include <mutex>
#include <memory>
#include "utils/mapPoint.h"
#include "utils/numType.h"

class MapPoint;

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

public:
    Frame();
    Frame(Mat rawImage, Mat rawDepth, Mat rawIntensity);

    Mat getDepth();
    Mat getImage();
    Mat getIntensity();

    void flagKeyFrame();
    bool isKeyframe() const;
    size_t getKeyframeId() const;

    SE3 getTrackingResult() const;
    SE3 getPoseInGlobalMap() const;
    SE3 getPoseInLocalMap() const;
    void setTrackingResult(const SE3 &T);
    void setRawKeyframePose(const SE3 &T);
    void setOptimizationResult(const SE3 &T);
    void setReferenceKF(std::shared_ptr<Frame> kf);
    std::shared_ptr<Frame> getReferenceKF() const;

    std::shared_ptr<Frame> referenceKF;

    bool inLocalOptimizer;
    double *getParameterBlock();

    // std::vector<Vec9f> pointDesc;
    Mat pointDesc;
    std::vector<float> depthVec;
    std::vector<cv::KeyPoint> cvKeyPoints;
    std::vector<std::shared_ptr<MapPoint>> mapPoints;

    void updateCovisibility();
    std::vector<std::shared_ptr<Frame>> getCovisibleKeyFrames(size_t th);
    std::vector<std::shared_ptr<Frame>> covisibleKFs;
};