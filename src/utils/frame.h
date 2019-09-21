#pragma once
#include <memory>
#include "utils/numType.h"

class Frame;

struct Point3D
{
    static size_t nextPtId;
    Point3D() : ptId(nextPtId++), inOptimizer(false) {}
    size_t ptId;
    bool visited;
    bool inOptimizer;
    Vec3d position;
    Vec9f descriptor;
    std::shared_ptr<Frame> hostKF;
};

class Frame
{
    Mat rawDepth;
    Mat rawImage;
    Mat rawIntensity;
    bool keyframeFlag;

    size_t kfId;
    static size_t nextKFId;

    // Raw pose update from the tracker, this stays unchanged.
    SE3 relativePose;
    // Raw world pose used for local map registeration. Only for KFs.
    SE3 rawKeyframePose;
    // stores result from the optimizer, highly volatile
    SE3 optimizedPose;

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

    std::vector<Vec9f> pointDesc;
    std::vector<cv::KeyPoint> cvKeyPoints;
    std::vector<std::shared_ptr<Point3D>> mapPoints;
};