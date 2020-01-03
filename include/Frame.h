#pragma once

#include <ORBextractor.h>
#include <ORBVocabulary.h>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "KeyFrame.h"
#include "MapPoint.h"

class KeyFrame;
class MapPoint;

class Frame
{
public:
    Frame() = default;

    Frame(const Frame &F);

    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &ts, const Eigen::Matrix3d &K, ORB_SLAM2::ORBextractor *extractor, ORB_SLAM2::ORBVocabulary *voc);

    void ExtractORB();

    void SetPose(const cv::Mat &Tcw);

    bool IsInFrustum(MapPoint *pMP, float viewingCosLimit);

public:
    void ExtractORB(const cv::Mat &imGray);

    void ComputeDepth(const cv::Mat &imDepth);

    // Frame ID
    unsigned long mnId;
    static unsigned long mnNextId;

    cv::Mat mImGray, mImDepth;

    // Frame timestamp
    double mTimeStamp;

    // Feature extractor
    ORB_SLAM2::ORBextractor *mpORBextractor;

    // Vocabulary used for relocalization.
    ORB_SLAM2::ORBVocabulary *mpORBvocabulary;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Reference Keyframe.
    KeyFrame *mpReferenceKF;

    // Calibration matrix
    Eigen::Matrix3d mK;
    static int width;
    static int height;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    static bool mbInitialized;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // Key points and descriptors
    int N;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvObsKeys;
    std::vector<bool> mvbOutlier;
    cv::Mat mDescriptors;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<MapPoint *> mvpMapPoints;
    std::vector<MapPoint *> mvObsMapPoints;

    // Pose
    Sophus::SE3d mTcw;

    int mObs;
};
