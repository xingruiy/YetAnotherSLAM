#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "MapPoint.h"
#include "KeyFrame.h"

namespace SLAM
{

class Matcher
{
public:
    Matcher(float nnratio = 0.6, bool checkOri = true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int SearchByProjection(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th = 3);

public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    float RadiusByViewingCos(const float &viewCos);

    float mfNNratio;
    bool mbCheckOrientation;
};

} // namespace SLAM