#include "utils/frame.h"

size_t Frame::nextKFId = 0;

Frame::Frame()
{
}

Frame::Frame(Mat rawImage, Mat rawDepth, Mat rawIntensity)
    : kfId(0)
{
    rawImage.copyTo(this->rawImage);
    rawDepth.copyTo(this->rawDepth);
    rawIntensity.copyTo(this->rawIntensity);
}

Mat Frame::getDepth()
{
    return rawDepth;
}

Mat Frame::getImage()
{
    return rawImage;
}

Mat Frame::getIntensity()
{
    return rawIntensity;
}

SE3 Frame::getPose()
{
    return framePose;
}

void Frame::setPose(const SE3 &T)
{
    framePose = T;
}

void Frame::flagKeyFrame()
{
    kfId = nextKFId++;
}