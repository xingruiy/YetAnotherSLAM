#pragma once
#include <memory>
#include "utils/frame.h"
#include "utils/numType.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

enum PointType
{
    FAST,
    ORB,
    HARRIS
};

class FeatureMatcher
{
    PointType pointType;
    cv::Ptr<cv::ORB> orbDetector;
    cv::Ptr<cv::FastFeatureDetector> fastDetector;

public:
    FeatureMatcher(PointType pType);
    void detect(
        Mat image, Mat depth,
        std::vector<cv::KeyPoint> &keyPoints,
        std::vector<Vec9f> &patch3x3,
        std::vector<float> &depthVec);
    void matchByProjection(
        std::shared_ptr<Frame> reference,
        std::shared_ptr<Frame> current,
        SE3 &Transform,
        std::vector<cv::DMatch> &matches);
};