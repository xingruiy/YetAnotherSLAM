#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "MapPoint.h"
#include "KeyFrame.h"

namespace SLAM
{

class ORBMatcher
{
public:
    ORBMatcher(float nnratio = 0.6, bool checkOri = true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int SearchByProjection(KeyFrame *pKF, const std::vector<MapPoint *> &vpMapPoints, const float th = 3);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(KeyFrame *pKF, const std::vector<MapPoint *> &vpMapPoints, const float th = 3.0);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                               std::vector<std::pair<size_t, size_t>> &vMatchedPairs,
                               const bool bOnlyStereo);

public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    float RadiusByViewingCos(const float &viewCos);
    void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);
    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF2);

    float mfNNratio;
    bool mbCheckOrientation;
};

} // namespace SLAM