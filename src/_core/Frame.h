#pragma once
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <DBoW2/DBoW2/BowVector.h>

#include "GlobalSettings.h"
#include "ORBVocabulary.h"

namespace slam
{

class MapPoint;
class KeyFrame;
class ORBextractor;

struct Intrinsics
{
    int w, h;
    float ifx, ify;
    float fx, fy, cx, cy;
};

struct FrameMetaData
{
    int id;
    double timestamp;

    FrameMetaData *ref;
    Sophus::SE3d camToRef;
};

class Frame
{
public:
    Frame();
    Frame(const Frame &frame);
    Frame(cv::Mat img, cv::Mat depth, ORBextractor *ext, ORBVocabulary *voc);
    int detectFeaturesInFrame();
    void ComputeBoW();
    bool isInFrustum(MapPoint *pMP, float viewingCosLimit);
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel = -1, const int maxLevel = -1) const;
    void ComputeStereoFromRGBD();
    void CreateRelocalisationPoints();

public:
    ORBVocabulary *OrbVoc;
    ORBextractor *OrbExt;
    FrameMetaData *meta;

    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;
    float mbf;
    float mb;
    float mThDepth;
    int N;

    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors;

    // MapPoints associated to keypoints, nullptr pointer if no association.
    std::vector<MapPoint *> mvpMapPoints;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid
    // to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    Sophus::SE3d mTcw;
    Sophus::SE3d mTcp;

    // Reference Keyframe.
    KeyFrame *mpReferenceKF;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;
    static bool mbInitialComputations;

    // A copy of the input frame
    cv::Mat mImGray;
    cv::Mat mImDepth;
    std::vector<Eigen::Vector3d> mvRelocPoints;

private:
    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();
};

} // namespace slam