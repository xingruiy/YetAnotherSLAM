#include "featureMatcher.h"

FeatureMatcher::FeatureMatcher(PointType pType)
    : pointType(pType)
{
    switch (pType)
    {
    case PointType::ORB:
        orbDetector = cv::ORB::create(1000);
        break;
    case PointType::FAST:
        fastDetector = cv::FastFeatureDetector::create();
        break;
    }
}

void FeatureMatcher::detect(
    Mat image, Mat depth,
    std::vector<cv::KeyPoint> &keyPoints,
    std::vector<Vec9f> &patch3x3,
    std::vector<float> &depthVec)
{
}

void FeatureMatcher::matchByProjection(
    std::shared_ptr<Frame> reference,
    std::shared_ptr<Frame> current,
    SE3 &Transform,
    std::vector<cv::DMatch> &matches)
{
}