#include "utils/frame.h"

Frame::Frame()
{
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