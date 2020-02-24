#pragma once
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <DBoW2/DBoW2/BowVector.h>

#include "KeyFrame.h"
#include "GlobalDef.h"

namespace SLAM
{

class KeyFrame;
// class MapPoint;

struct Frame
{
    ~Frame();
    Frame() = default;
    Frame(const Frame &F);
    Frame(cv::Mat image, cv::Mat depth, double timeStamp);

    // Frame ID
    unsigned long mnId;
    static unsigned long mnNextId;

    // Pose
    Sophus::SE3d mTcw;
    Sophus::SE3d T_frame2Ref;

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
};

} // namespace SLAM