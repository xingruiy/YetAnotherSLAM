#pragma once
#include <memory>
#include "utils/numType.h"

class Frame;

struct Point3D
{
    Point3D() : ptId(nextPtId++), inOptimizer(false) {}
    size_t ptId;
    static size_t nextPtId;
    bool visited;
    bool inOptimizer;
    Vec3d position;
    Vec9f descriptor;
    std::shared_ptr<Frame> hostKF;
    std::vector<std::shared_ptr<Frame>> frameHistory;
};

class Frame
{
    Mat rawDepth;
    Mat rawImage;
    Mat rawIntensity;
    SE3 framePose;
    static size_t nextKFId;

    std::vector<std::shared_ptr<Point3D>> worldPoints;

public:
    Frame();
    Frame(Mat rawImage, Mat rawDepth, Mat rawIntensity);
    Mat getDepth();
    Mat getImage();
    Mat getIntensity();
    SE3 getPose();
    void flagKeyFrame();
    void setPose(const SE3 &T);

    size_t kfId;
    SE3 Tr2c;
    std::shared_ptr<Frame> referenceKF;
    std::vector<Vec9f> pointDesc;
    std::vector<cv::KeyPoint> cvKeyPoints;
    std::vector<std::shared_ptr<Point3D>> mapPoints;
};