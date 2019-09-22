#include "utils/frame.h"

size_t Frame::nextKFId = 0;
size_t Point3D::nextPtId = 0;

Frame::Frame()
{
}

Frame::Frame(Mat rawImage, Mat rawDepth, Mat rawIntensity)
    : kfId(0), keyframeFlag(false)
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

void Frame::flagKeyFrame()
{
    keyframeFlag = true;
    kfId = nextKFId++;
    // referenceKF = NULL;
    // relativePose = SE3();
}

bool Frame::isKeyframe() const
{
    return keyframeFlag;
}

size_t Frame::getKeyframeId() const
{
    return kfId;
}

SE3 Frame::getPoseInGlobalMap() const
{
    if (isKeyframe())
        return optimizedPose;
    else
    {
        auto referencePose = referenceKF->getPoseInGlobalMap();
        return referencePose * relativePose;
    }
}

SE3 Frame::getTrackingResult() const
{
    return relativePose;
}

SE3 Frame::getPoseInLocalMap() const
{
    if (isKeyframe())
        return rawKeyframePose;
    else
    {
        auto referencePose = referenceKF->getPoseInLocalMap();
        return referencePose * relativePose;
    }
}

void Frame::setReferenceKF(std::shared_ptr<Frame> kf)
{
    referenceKF = kf;
}

std::shared_ptr<Frame> Frame::getReferenceKF() const
{
    return referenceKF;
}

void Frame::setTrackingResult(const SE3 &T)
{
    relativePose = T;
}

void Frame::setOptimizationResult(const SE3 &T)
{
    if (isKeyframe())
        optimizedPose = T;
}

void Frame::setRawKeyframePose(const SE3 &T)
{
    optimizedPose = T;
    rawKeyframePose = T;
}