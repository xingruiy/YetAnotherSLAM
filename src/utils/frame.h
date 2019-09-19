#pragma once
#include "utils/numType.h"

class Frame
{
    Mat rawDepth;
    Mat rawImage;
    Mat rawIntensity;
    SE3 framePose;

public:
    Frame();
    Frame(Mat rawImage, Mat rawDepth, Mat rawIntensity);
    Mat getDepth();
    Mat getImage();
    Mat getIntensity();
    SE3 getPose();
    void setPose(const SE3 &T);
};