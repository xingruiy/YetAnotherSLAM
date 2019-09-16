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
    Mat getDepth();
    Mat getImage();
    Mat getIntensity();
    SE3 getPose();
};