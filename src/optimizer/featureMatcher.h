#pragma once
#include <memory>
#include "utils/frame.h"
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

    void computePatch3x3(
        Mat image,
        std::vector<cv::KeyPoint> &points,
        std::vector<Vec9f> &patches);

    void extractDepth(
        Mat depth,
        std::vector<cv::KeyPoint> &points,
        std::vector<float> &depthVec);

public:
    FeatureMatcher(PointType pType, DescType dType);

    void detect(
        Mat image, Mat depth,
        std::vector<cv::KeyPoint> &keyPoints,
        Mat &descriptor,
        std::vector<float> &depthVec);

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
        const std::vector<std::shared_ptr<MapPoint>> mapPoints,
        const std::shared_ptr<Frame> frame,
        const Mat33d &K,
        std::vector<cv::DMatch> &matches,
        std::vector<bool> *matchesFound = NULL);

    void matchByDescriptor(
        const std::shared_ptr<Frame> kf,
        const std::shared_ptr<Frame> frame,
        const Mat33d &K,
        std::vector<cv::DMatch> &matches);

    void compute(
        Mat image,
        std::vector<cv::KeyPoint> pt,
        Mat &desc);

    float computeMatchingScore(
        Mat desc,
        Mat refDesc);
};