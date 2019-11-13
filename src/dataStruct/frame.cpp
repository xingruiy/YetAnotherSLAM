#include "dataStruct/frame.h"

size_t Frame::nextKFId = 0;

Frame::Frame()
{
}

Frame::Frame(Mat imRGB, Mat imDepth, Mat imGray, Mat33d &K)
    : kfId(0), keyframeFlag(false), K(K)
{
    imRGB.copyTo(this->imRGB);
    imDepth.copyTo(this->imDepth);
    imDepth.copyTo(ogDepth);
    imGray.copyTo(rawIntensity);
}

Mat33d Frame::getIntrinsics() const
{
    return K;
}

Mat Frame::getDepth() const
{
    return imDepth;
}

Mat Frame::getImage() const
{
    return imRGB;
}

Mat Frame::getIntensity() const
{
    return rawIntensity;
}

Mat Frame::getOGDepth() const
{
    return ogDepth;
}

void Frame::flagKeyFrame()
{
    keyframeFlag = true;
    kfId = nextKFId++;
}

bool Frame::isKeyframe() const
{
    return keyframeFlag;
}

size_t Frame::getId() const
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

Vec3d Frame::getPositionWorld() const
{
    SE3 T = getPoseInGlobalMap();
    return T.translation();
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

void Frame::setReferenceKF(std::shared_ptr<Frame> kf)
{
    referenceKF = kf;
}

std::shared_ptr<Frame> Frame::getReferenceKF() const
{
    return referenceKF;
}

double *Frame::getParameterBlock()
{
    return optimizedPose.data();
}

bool Frame::hasMapPoint() const
{
    return !cvKeyPoints.empty();
}

void Frame::setMapPoint(std::shared_ptr<MapPoint> pt, size_t idx)
{
    mapPoints[idx] = pt;
}

void Frame::eraseMapPoint(size_t idx)
{
    mapPoints[idx] = NULL;
}

const std::vector<std::shared_ptr<MapPoint>> &Frame::getMapPoints() const
{
    return mapPoints;
}

void Frame::detectKeyPoints(std::shared_ptr<FeatureMatcher> matcher)
{
    if (cvKeyPoints.size() == 0)
    {
        matcher->detectAndCompute(imRGB, cvKeyPoints, descriptors);
        matcher->computePointDepth(imDepth, cvKeyPoints, keyPointDepth);
        matcher->computePointNormal(nmap, cvKeyPoints, keyPointNorm);
        mapPoints.resize(cvKeyPoints.size());
    }
}

std::shared_ptr<MapPoint> Frame::createMapPoint(size_t idx)
{
    const auto z = keyPointDepth[idx];
    if (z > FLT_EPSILON)
    {
        auto pt = std::make_shared<MapPoint>();
        const auto &kp = cvKeyPoints[idx].pt;
        const auto &desc = descriptors.row(idx);
        pt->setDescriptor(desc);
        pt->setPosWorld(optimizedPose * (K.inverse() * Vec3d(kp.x, kp.y, 1.0) * z));
        return pt;
    }

    return NULL;
}

Mat Frame::getNormalMap() const
{
    return nmap;
}

void Frame::setNormalMap(const Mat nmap)
{
    this->nmap = nmap;
}