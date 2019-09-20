#pragma once
#include <memory>
#include "utils/numType.h"

class Frame;

struct Point3D
{
    bool visited;
    Vec3d position;
    Vec9f descriptor;
    std::shared_ptr<Frame> hostKF;
    std::vector<std::pair<std::shared_ptr<Frame>, Vec2d>> observations;
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

    std::vector<Vec9f> pointDesc;
    std::vector<cv::KeyPoint> cvKeyPoints;
    std::vector<std::shared_ptr<Point3D>> mapPoints;
};