#pragma once
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <DBoW2/DBoW2/BowVector.h>

#include "MapPoint.h"
#include "KeyFrame.h"
#include "GlobalDef.h"
#include "ORBextractor.h"
#include "ORBVocabulary.h"

namespace SLAM
{

class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat imGray, const cv::Mat imDepth, const double &timeStamp, ORBextractor *extractor);

    // Extract ORB on the image.
    void ExtractORB();

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(const Sophus::SE3d &Tcw);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

public:
    // Vocabulary used for relocalization.
    ORBVocabulary *mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor *mpORBextractor;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of KeyPoints.
    int N;

    // Vector of keypoints (original for visualization)
    // and undistorted (actually used by the system).
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors;

    // MapPoints associated to keypoints, NULL pointer if no association.
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
    Sophus::SE3d mRelativePose;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

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

    // A copy of the input frame
    cv::Mat mImGray;
    cv::Mat mImDepth;

    static bool mbInitialComputations;

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

} // namespace SLAM