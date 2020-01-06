#pragma once
#include "Map.h"
#include "Frame.h"
#include "MapPoint.h"
#include "ORBVocabulary.h"
#include <memory>

class Map;
class Frame;
class MapPoint;

class KeyFrame
{
public:
    KeyFrame(const Frame &F, Map *pMap);

    // Bag of Words Representation
    void ComputeBoW();

    bool IsInFrustum(MapPoint *pMP, float viewingCosLimit);

    std::vector<MapPoint *> GetMapPointMatches();
    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const;

public:
    unsigned long mnId;
    static unsigned long nNextId;
    double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> mGrid;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Calibration parameters
    const int width, height;
    const float mbf, mThDepth;
    const float fx, fy, cx, cy, invfx, invfy;

    // Number of KeyPoints
    const int N;

    // KeyPoints
    const std::vector<float> mvDepth;
    const std::vector<float> mvuRight;
    const std::vector<cv::KeyPoint> mvKeys;
    const cv::Mat mDescriptors;

    // MapPoints associated to keypoints
    std::vector<bool> mvbOutlier;
    std::vector<MapPoint *> mvpMapPoints;
    std::vector<MapPoint *> mvpParentMPs;

    // BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    ORB_SLAM2::ORBVocabulary *mpORBvocabulary;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    Sophus::SE3d mTcw;
    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;

    Map *mpMap;
    KeyFrame *mpParent;
};