#pragma once
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <DBoW2/DBoW2/BowVector.h>

#include "MapPoint.h"
#include "KeyFrame.h"
#include "GlobalDef.h"
#include "ORBextractor.h"

namespace SLAM
{

class KeyFrame;
// class MapPoint;

struct Frame
{
    ~Frame();
    Frame() = default;
    Frame(const Frame &F);
    Frame(cv::Mat image, cv::Mat depth, double timeStamp, ORBextractor *pExtractor);

    void ExtractORBFeatures();
    void ComputeBoW(ORB_SLAM2::ORBVocabulary *voc);

    // Frame ID
    unsigned long mnId;
    static unsigned long mnNextId;

    // Pose
    Sophus::SE3d mTcw;
    Sophus::SE3d mRelativePose;

    // A copy of the input frame
    cv::Mat mImGray;
    cv::Mat mImDepth;

    // Frame timestamp
    double mTimeStamp;

    // Reference Keyframe.
    KeyFrame *mpReferenceKF;

    bool mbIsKeyFrame;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    int N;
    std::vector<float> mvfDepth;
    std::vector<bool> mvbOutliers;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<MapPoint *> mvMapPoints;
    cv::Mat mDescriptors;
    ORBextractor *mpORBExtractor;
};

} // namespace SLAM