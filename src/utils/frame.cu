#include "utils/frame.h"

Frame::Frame()
{
}

Frame::Frame(Mat rawImage, Mat rawDepth)
{
    rawImage.copyTo(this->rawImage);
    rawDepth.copyTo(this->rawDepth);
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