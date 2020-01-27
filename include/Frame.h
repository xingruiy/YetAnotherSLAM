#pragma once

#include <ORBextractor.h>
#include <ORBVocabulary.h>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "KeyFrame.h"
#include "MapPoint.h"

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class KeyFrame;
class MapPoint;

class Frame
{
public:
    Frame() = default;

    Frame(const Frame &F);

    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &ts, const Eigen::Matrix3d &K,
          const float &bf, const float &thDepth, cv::Mat &distCoef, ORB_SLAM2::ORBextractor *extractor,
          ORB_SLAM2::ORBVocabulary *voc);

    void ExtractORB();
    bool IsInFrustum(MapPoint *pMP, float viewingCosLimit);

public:
    void ExtractORB(const cv::Mat &imGray);
    void ComputeDepth(const cv::Mat &imDepth);
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    void ComputeImageBounds(const cv::Mat &img);
    void UndistortKeyPoints();
    void AssignFeaturesToGrid();

    // Pose
    Sophus::SE3d mTcw;

    // Frame ID
    unsigned long mnId;
    static unsigned long mnNextId;

    // A copy of the input frame
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
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;
    float mThDepth;

    static bool mbInitialized;
    // Camera parameters
    static float fx, fy, cx, cy, invfx, invfy;

    // Image bounds
    static float mnMinX, mnMinY, mnMaxX, mnMaxY;

    // Keypoints are assigned to cells in a grid
    // to reduce matching complexity when projecting MapPoints.
    vector<size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // Total number of key points
    int N;
    // Feature descriptors
    cv::Mat mDescriptors;

    // Original and undistorted key points
    std::vector<bool> mvbOutlier;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate
    std::vector<float> mvuRight;

    // Depth for each keypoint.
    std::vector<float> mvDepth;

    // MapPoints associated to keypoints
    // NULL pointer if no association.
    std::vector<MapPoint *> mvpMapPoints;
};
