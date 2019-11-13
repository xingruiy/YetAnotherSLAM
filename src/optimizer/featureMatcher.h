#pragma once
#include <memory>
#include "dataStruct/frame.h"
#include "utils/numType.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

class Frame;
class MapPoint;

enum class PointType
{
    FAST,
    ORB,
    BRISK,
    SURF
};

enum class DescType
{
    ORB,
    BRISK,
    SURF
};

class FeatureMatcher
{
    PointType pointType;
    DescType descType;
    float minMatchingDistance;
    cv::Ptr<cv::ORB> orbDetector;
    cv::Ptr<cv::BRISK> briskDetector;
    cv::Ptr<cv::xfeatures2d::SURF> surfDetector;
    cv::Ptr<cv::FastFeatureDetector> fastDetector;
    cv::Ptr<cv::DescriptorMatcher> matcher;

public:
    FeatureMatcher(PointType pType, DescType dType);

    void matchByProjection(
        const std::shared_ptr<Frame> kf,
        const std::shared_ptr<Frame> frame,
        const Mat33d &K,
        std::vector<cv::DMatch> &matches,
        std::vector<bool> *matchesFound = NULL);

    void matchByProjection2NN(
        const std::shared_ptr<Frame> kf,
        const std::shared_ptr<Frame> frame,
        const Mat33d &K,
        std::vector<cv::DMatch> &matches,
        std::vector<bool> *matchesFound = NULL);

    void matchByProjection2NN(
        const std::vector<std::shared_ptr<MapPoint>> &mapPoints,
        const std::shared_ptr<Frame> frame,
        const Mat33d &K,
        std::vector<cv::DMatch> &matches,
        std::vector<bool> *matchesFound = NULL);

    float computeMatchingScore(
        Mat desc,
        Mat refDesc);

    // Feature Detection
    void detectAndCompute(
        const Mat image,
        std::vector<cv::KeyPoint> &cvKeyPoint,
        Mat &descriptors);

    void computePointDepth(
        const Mat depth,
        const std::vector<cv::KeyPoint> &cvKeyPoint,
        std::vector<float> &pointDepth);

    void computePointNormal(
        const Mat normal,
        const std::vector<cv::KeyPoint> &cvKeyPoint,
        std::vector<Vec3f> &pointNormal);
};