#ifndef _POSE_SOLVER_H
#define _POSE_SOLVER_H

#include <opencv2/opencv.hpp>
#include <sophus/sim3.hpp>
#include <vector>

#include "KeyFrame.h"

namespace slam
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

} // namespace slam

#endif // SIM3SOLVER_H
