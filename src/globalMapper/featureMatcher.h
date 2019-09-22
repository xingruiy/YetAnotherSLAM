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
    FeatureMatcher(PointType pType);

    void detect(
        Mat image, Mat depth, Mat intensity,
        std::vector<cv::KeyPoint> &keyPoints,
        std::vector<Vec9f> &patch3x3,
        std::vector<float> &depthVec);

    void matchByProjection(
        const std::shared_ptr<Frame> kf,
        const std::shared_ptr<Frame> frame,
        const Mat33d &K,
        std::vector<cv::DMatch> &matches,
        std::vector<bool> *matchesFound = NULL);

    void compute(
        Mat image,
        std::vector<cv::KeyPoint> pt,
        Mat &desc);

    float computeMatchingScore(
        Mat desc,
        Mat refDesc);
};