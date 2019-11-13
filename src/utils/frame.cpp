#include "utils/frame.h"

size_t Frame::nextKFId = 0;

Frame::Frame()
{
}

Frame::Frame(int w,
             int h,
             Mat33d &K,
             Mat colourImage,
             Mat depthImage,
             Mat intensityImage)
    : kfId(0),
      numPointsDetectd(0),
      numPointsCreated(0),
      imgWidth(w),
      imgHeight(h),
      keyframeFlag(false),
      camIntrinsics(K)
{
    colourImage.copyTo(rawImage);
    depthImage.copyTo(rawDepth);
    depthImage.copyTo(ogDepth);
    intensityImage.copyTo(rawIntensity);
}

int Frame::getImageWidth() const
{
    return imgWidth;
}

int Frame::getImageHeight() const
{
    return imgHeight;
}

Mat33d Frame::getIntrinsics() const
{
    return camIntrinsics;
}

Mat Frame::getDepth() const
{
    return rawDepth;
}

Mat Frame::getImage() const
{
    return rawImage;
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

size_t Frame::getNumPointsDetected() const
{
    return numPointsDetectd;
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
    numPointsCreated != 0;
}

void Frame::setMapPoint(std::shared_ptr<MapPoint> pt, size_t idx)
{
    mapPoints[idx] = pt;
    numPointsCreated++;
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
    if (numPointsDetectd == 0)
    {
        matcher->detectAndCompute(rawImage, cvKeyPoints, pointDesc);
        matcher->computePointDepth(rawDepth, cvKeyPoints, keyPointDepth);
        matcher->computePointNormal(normalMap, cvKeyPoints, keyPointNorm);
        numPointsDetectd = cvKeyPoints.size();
        mapPoints.resize(numPointsDetectd);
    }
}

std::shared_ptr<MapPoint> Frame::createMapPoint(size_t idx)
{
    const auto z = keyPointDepth[idx];
    if (z > FLT_EPSILON)
    {
        auto pt = std::make_shared<MapPoint>();
        const auto &kp = cvKeyPoints[idx].pt;
        const auto &desc = pointDesc.row(idx);
        pt->setDescriptor(desc);
        pt->setPosWorld(optimizedPose * (camIntrinsics.inverse() * Vec3d(kp.x, kp.y, 1.0) * z));
        return pt;
    }

    return NULL;
}

Mat Frame::getNormalMap() const
{
    return normalMap;
}

void Frame::setNormalMap(const Mat nmap)
{
    normalMap = nmap;
}