#include "utils/frame.h"

Frame::Frame()
{
}

Frame::Frame(Mat rawImage, Mat rawDepth, Mat rawIntensity)
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

void Frame::minimizeFootPrint()
{
    rawDepth.release();
    rawIntensity.release();
}

size_t Frame::getKeyPointSize() const
{
    return worldPoints.size();
}

std::vector<std::shared_ptr<PointWorld>> Frame::getWorldPoints() const
{
    return worldPoints;
}

void Frame::setWorldPoints(const std::vector<std::shared_ptr<PointWorld>> &pt)
{
    worldPoints = pt;
}