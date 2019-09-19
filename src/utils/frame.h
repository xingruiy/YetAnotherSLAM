#pragma once
#include <memory>
#include "utils/numType.h"

class Frame;

struct PointWorld
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

    std::vector<std::shared_ptr<PointWorld>> worldPoints;

public:
    Frame();
    Frame(Mat rawImage, Mat rawDepth, Mat rawIntensity);
    Mat getDepth();
    Mat getImage();
    Mat getIntensity();
    SE3 getPose();
    void setPose(const SE3 &T);
    void minimizeFootPrint();
    size_t getKeyPointSize() const;
    std::vector<std::shared_ptr<PointWorld>> getWorldPoints() const;
    void setWorldPoints(const std::vector<std::shared_ptr<PointWorld>> &pt);
};