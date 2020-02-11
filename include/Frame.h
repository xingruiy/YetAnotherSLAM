#pragma once

#include <ORBextractor.h>
#include <ORBVocabulary.h>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "KeyFrame.h"
#include "MapPoint.h"
#include "GlobalDef.h"

namespace SLAM
{

class KeyFrame;
class MapPoint;

class Frame
{
public:
    Frame() = default;

    Frame(const Frame &F);

    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &ts,
          ORB_SLAM2::ORBextractor *extractor, ORB_SLAM2::ORBVocabulary *voc);

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
    std::vector<size_t> orbGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

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
    cv::Mat mDescriptors;
    // Original and undistorted key points
    std::vector<bool> mvbOutlier;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;
    std::vector<MapPoint *> mvpMapPoints;
};

} // namespace SLAM