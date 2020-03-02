/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>
#include <sophus/sim3.hpp>
#include <opencv2/core/core.hpp>

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

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    int SearchByProjection(KeyFrame *pKF, Sophus::SE3d Scw, const std::vector<MapPoint *> &vpPoints, std::vector<MapPoint *> &vpMatched, int th);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                               std::vector<std::pair<size_t, size_t>> &vMatchedPairs,
                               const bool bOnlyStereo);

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    int SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::SE3d &S12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(KeyFrame *pKF, const std::vector<MapPoint *> &vpMapPoints, const float th = 3.0);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(KeyFrame *pKF, Sophus::SE3d Tcw, const std::vector<MapPoint *> &vpPoints, float th, std::vector<MapPoint *> &vpReplacePoint);

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

#endif