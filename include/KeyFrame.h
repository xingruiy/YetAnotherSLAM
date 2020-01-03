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

public:
    unsigned long mnId;
    static unsigned long nNextId;
    double mTimeStamp;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy;

    // Number of KeyPoints
    const int N;

    // KeyPoints
    const std::vector<float> mvDepth;
    const std::vector<cv::KeyPoint> mvKeys;
    const cv::Mat mDescriptors;

    // MapPoints associated to keypoints
    std::vector<bool> mvbOutlier;
    std::vector<MapPoint *> mvpMapPoints;

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

    Map *mpMap;

    std::mutex mMutexPose;
};