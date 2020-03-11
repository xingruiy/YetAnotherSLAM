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

#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <sophus/sim3.hpp>
#include <vector>

#include "KeyFrame.h"

namespace SLAM
{

class PoseSolver
{
public:
    PoseSolver(Frame *pFrame, KeyFrame *pKF2, const std::vector<MapPoint *> &vpMatched12);
    PoseSolver(KeyFrame *pKF1, KeyFrame *pKF2, const std::vector<MapPoint *> &vpMatched12, const bool bFixScale = true);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6, int maxIterations = 300);

    bool find(std::vector<bool> &vbInliers12, int &nInliers, Sophus::SE3d &Scw);

    bool iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers, Sophus::SE3d &Scw);

    Eigen::Matrix3d GetEstimatedRotation();
    Eigen::Vector3d GetEstimatedTranslation();
    float GetEstimatedScale();

protected:
    void ComputeCentroid(const std::vector<Eigen::Vector3d> &P, std::vector<Eigen::Vector3d> &Pr, Eigen::Vector3d &C);

    void ComputeSE3(const std::vector<Eigen::Vector3d> &P1, const std::vector<Eigen::Vector3d> &P2);

    void CheckInliers();

    void Project(const std::vector<Eigen::Vector3d> &vP3Dw, std::vector<Eigen::Vector2d> &vP2D, Sophus::SE3d Twc, cv::Mat K);
    void FromCameraToImage(const std::vector<Eigen::Vector3d> &vP3Dc, std::vector<Eigen::Vector2d> &vP2D, cv::Mat K);

protected:
    // KeyFrame matches
    std::vector<Eigen::Vector3d> mvpFramePoints;

    std::vector<Eigen::Vector3d> mvX3Dc1;
    std::vector<Eigen::Vector3d> mvX3Dc2;
    std::vector<MapPoint *> mvpMapPoints1;
    std::vector<MapPoint *> mvpMapPoints2;
    std::vector<MapPoint *> mvpMatches12;
    std::vector<size_t> mvnIndices1;
    std::vector<size_t> mvSigmaSquare1;
    std::vector<size_t> mvSigmaSquare2;
    std::vector<size_t> mvnMaxError1;
    std::vector<size_t> mvnMaxError2;

    int N;
    int mN1;

    // Current Estimation
    Eigen::Matrix3d mR12i;
    Eigen::Vector3d mt12i;
    float ms12i;
    Sophus::SE3d mT12i;
    Sophus::SE3d mT21i;
    std::vector<bool> mvbInliersi;
    int mnInliersi;

    // Current Ransac State
    int mnIterations;
    std::vector<bool> mvbBestInliers;
    int mnBestInliers;
    Sophus::SE3d mBestT12;
    Eigen::Matrix3d mBestRotation;
    Eigen::Vector3d mBestTranslation;
    float mBestScale;

    // Scale is fixed to 1 in the stereo/RGBD case
    bool mbFixScale;

    // Indices for random selection
    std::vector<size_t> mvAllIndices;

    // Projections
    std::vector<Eigen::Vector2d> mvP1im1;
    std::vector<Eigen::Vector2d> mvP2im2;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;

    // RANSAC max iterations
    int mRansacMaxIts;

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;
    float mSigma2;

    // Calibration
    cv::Mat mK1;
    cv::Mat mK2;
};

} // namespace SLAM

#endif // SIM3SOLVER_H
