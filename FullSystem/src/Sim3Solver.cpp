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

#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "ORBMatcher.h"

#include "DBoW2/DUtils/Random.h"

namespace SLAM
{

Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const std::vector<MapPoint *> &vpMatched12, const bool bFixScale)
    : mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
{
    mpKF1 = pKF1;
    mpKF2 = pKF2;

    auto vpKeyFrameMP1 = pKF1->GetMapPointMatches();

    mN1 = vpMatched12.size();

    mvpMapPoints1.reserve(mN1);
    mvpMapPoints2.reserve(mN1);
    mvpMatches12 = vpMatched12;
    mvnIndices1.reserve(mN1);
    mvX3Dc1.reserve(mN1);
    mvX3Dc2.reserve(mN1);

    const Sophus::SE3d &Twc1 = pKF1->mTcw.inverse();
    const Sophus::SE3d &Twc2 = pKF2->mTcw.inverse();

    mvAllIndices.reserve(mN1);

    size_t idx = 0;
    for (int i1 = 0; i1 < mN1; i1++)
    {
        if (vpMatched12[i1])
        {
            MapPoint *pMP1 = vpKeyFrameMP1[i1];
            MapPoint *pMP2 = vpMatched12[i1];

            if (!pMP1)
                continue;

            if (pMP1->isBad() || pMP2->isBad())
                continue;

            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);
            int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);

            if (indexKF1 < 0 || indexKF2 < 0)
                continue;

            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];

            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

            mvnMaxError1.push_back(9.210 * sigmaSquare1);
            mvnMaxError2.push_back(9.210 * sigmaSquare2);

            mvpMapPoints1.push_back(pMP1);
            mvpMapPoints2.push_back(pMP2);
            mvnIndices1.push_back(i1);

            Eigen::Vector3d X3D1w = pMP1->GetWorldPos();
            mvX3Dc1.push_back(Twc1 * X3D1w);

            Eigen::Vector3d X3D2w = pMP2->GetWorldPos();
            mvX3Dc2.push_back(Twc2 * X3D2w);

            mvAllIndices.push_back(idx);
            idx++;
        }
    }

    mK1 = pKF1->mK;
    mK2 = pKF2->mK;

    FromCameraToImage(mvX3Dc1, mvP1im1, mK1);
    FromCameraToImage(mvX3Dc2, mvP2im2, mK2);

    SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;

    N = mvpMapPoints1.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers / N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if (mRansacMinInliers == N)
        nIterations = 1;
    else
        nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(epsilon, 3)));

    mRansacMaxIts = std::max(1, std::min(nIterations, mRansacMaxIts));

    mnIterations = 0;
}

bool Sim3Solver::iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers, Sophus::SE3d &Scw)
{
    bNoMore = false;
    vbInliers = std::vector<bool>(mN1, false);
    nInliers = 0;

    if (N < mRansacMinInliers)
    {
        bNoMore = true;
        return false;
    }

    std::vector<size_t> vAvailableIndices;

    std::vector<Eigen::Vector3d> P3Dc1i(3);
    std::vector<Eigen::Vector3d> P3Dc2i(3);

    int nCurrentIterations = 0;
    while (mnIterations < mRansacMaxIts && nCurrentIterations < nIterations)
    {
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        for (short i = 0; i < 3; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            P3Dc1i[i] = mvX3Dc1[idx];
            P3Dc2i[i] = mvX3Dc2[idx];

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        ComputeSim3(P3Dc1i, P3Dc2i);

        CheckInliers();

        if (mnInliersi >= mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i;
            mBestRotation = mR12i;
            mBestTranslation = mt12i;
            mBestScale = ms12i;

            if (mnInliersi > mRansacMinInliers)
            {
                nInliers = mnInliersi;
                for (int i = 0; i < N; i++)
                    if (mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;
                Scw = mBestT12;
                return true;
            }
        }
    }

    if (mnIterations >= mRansacMaxIts)
        bNoMore = true;

    return false;
}

bool Sim3Solver::find(std::vector<bool> &vbInliers12, int &nInliers, Sophus::SE3d &Scw)
{
    bool bFlag;
    return iterate(mRansacMaxIts, bFlag, vbInliers12, nInliers, Scw);
}

void Sim3Solver::ComputeCentroid(const std::vector<Eigen::Vector3d> &P, std::vector<Eigen::Vector3d> &Pr, Eigen::Vector3d &C)
{
    for (int i = 0; i < P.size(); ++i)
        C += P[i];
    C = C / P.size();

    for (int i = 0; i < P.size(); i++)
        Pr[i] = P[i] - C;
}

void Sim3Solver::ComputeSim3(const std::vector<Eigen::Vector3d> &P1, const std::vector<Eigen::Vector3d> &P2)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates

    // Relative coordinates to centroid (set 1)
    std::vector<Eigen::Vector3d> Pr1(P1.size());
    // Relative coordinates to centroid (set 2)
    std::vector<Eigen::Vector3d> Pr2(P2.size());
    // Centroid of P1
    Eigen::Vector3d O1 = Eigen::Vector3d::Zero();
    // Centroid of P2
    Eigen::Vector3d O2 = Eigen::Vector3d::Zero();

    ComputeCentroid(P1, Pr1, O1);
    ComputeCentroid(P2, Pr2, O2);

    // Step 2: Compute M matrix

    // cv::Mat M = Pr2 * Pr1.t();
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    for (int i = 0; i < Pr1.size(); ++i)
        M += Pr2[i] * Pr1[i].transpose();

    const auto SVD = M.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto MatU = SVD.matrixU();
    const auto MatV = SVD.matrixV();
    Eigen::Matrix3d R;

    //! Check if R is a valid rotation matrix
    if (MatU.determinant() * MatV.determinant() < 0)
    {
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        I(2, 2) = -1;
        R = MatV * I * MatU.transpose();
    }
    else
    {
        R = MatV * MatU.transpose();
    }

    const auto t = O1 - R * O2;
    mT12i = Sophus::SE3d(R, t);
    mT21i = mT12i.inverse();
    // Step 3: Compute N matrix

    // double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    // N11 = M(0, 0) + M(1, 1) + M(2, 2);
    // N12 = M(1, 2) - M(2, 1);
    // N13 = M(2, 0) - M(0, 2);
    // N14 = M(0, 1) - M(1, 0);
    // N22 = M(0, 0) - M(1, 1) - M(2, 2);
    // N23 = M(0, 1) + M(1, 0);
    // N24 = M(2, 0) + M(0, 2);
    // N33 = -M(0, 0) + M(1, 1) - M(2, 2);
    // N34 = M(1, 2) + M(2, 1);
    // N44 = -M(0, 0) - M(1, 1) + M(2, 2);

    // cv::Mat N = (cv::Mat_<float>(4, 4) << N11, N12, N13, N14,
    //              N12, N22, N23, N24,
    //              N13, N23, N33, N34,
    //              N14, N24, N34, N44);

    // // Step 4: Eigenvector of the highest eigenvalue

    // cv::Mat eval, evec;

    // cv::eigen(N, eval, evec); //evec[0] is the quaternion of the desired rotation

    // cv::Mat vec(1, 3, evec.type());
    // (evec.row(0).colRange(1, 4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

    // // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    // double ang = atan2(norm(vec), evec.at<float>(0, 0));

    // vec = 2 * ang * vec / norm(vec); //Angle-axis representation. quaternion angle is the half

    // cv::Mat aux_mR12i;
    // aux_mR12i.create(3, 3, vec.type());

    // cv::Rodrigues(vec, aux_mR12i); // computes the rotation matrix from angle-axis

    // mR12i << aux_mR12i.at<float>(0, 0), aux_mR12i.at<float>(0, 1), aux_mR12i.at<float>(0, 2),
    //     aux_mR12i.at<float>(1, 0), aux_mR12i.at<float>(1, 1), aux_mR12i.at<float>(1, 2),
    //     aux_mR12i.at<float>(2, 0), aux_mR12i.at<float>(2, 1), aux_mR12i.at<float>(2, 2);

    // // Step 5: Rotate set 2
    // // not required if we fix scales

    // // Step 6: Scale

    // ms12i = 1.0f;

    // // Step 7: Translation

    // mt12i = O1 - ms12i * mR12i * O2;

    // // Step 8: Transformation

    // Eigen::Matrix3d sR = ms12i * mR12i;
    // mT21i = Sophus::SE3d(sR, mt12i);
    // mT12i = mT21i.inverse();
}

void Sim3Solver::CheckInliers()
{
    std::vector<Eigen::Vector2d> vP1im2, vP2im1;
    Project(mvX3Dc2, vP2im1, mT12i, mK1);
    Project(mvX3Dc1, vP1im2, mT21i, mK2);

    mnInliersi = 0;

    for (size_t i = 0; i < mvP1im1.size(); i++)
    {
        Eigen::Vector2d dist1 = mvP1im1[i] - vP2im1[i];
        Eigen::Vector2d dist2 = vP1im2[i] - mvP2im2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if (err1 < mvnMaxError1[i] && err2 < mvnMaxError2[i])
        {
            mvbInliersi[i] = true;
            mnInliersi++;
        }
        else
            mvbInliersi[i] = false;
    }
}

Eigen::Matrix3d Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation;
}

Eigen::Vector3d Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation;
}

float Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}

void Sim3Solver::Project(const std::vector<Eigen::Vector3d> &vP3Dw, std::vector<Eigen::Vector2d> &vP2D, Sophus::SE3d Twc, cv::Mat K)
{
    const float &fx = K.at<float>(0, 0);
    const float &fy = K.at<float>(1, 1);
    const float &cx = K.at<float>(0, 2);
    const float &cy = K.at<float>(1, 2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for (size_t i = 0, iend = vP3Dw.size(); i < iend; i++)
    {
        Eigen::Vector3d P3Dc = Twc * vP3Dw[i];
        const float invz = 1 / (P3Dc(2));
        const float x = P3Dc(0) * invz;
        const float y = P3Dc(1) * invz;

        vP2D.push_back(Eigen::Vector2d(fx * x + cx, fy * y + cy));
    }
}

void Sim3Solver::FromCameraToImage(const std::vector<Eigen::Vector3d> &vP3Dc, std::vector<Eigen::Vector2d> &vP2D, cv::Mat K)
{
    const float &fx = K.at<float>(0, 0);
    const float &fy = K.at<float>(1, 1);
    const float &cx = K.at<float>(0, 2);
    const float &cy = K.at<float>(1, 2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for (size_t i = 0, iend = vP3Dc.size(); i < iend; i++)
    {
        const float invz = 1 / vP3Dc[i](2);
        const float x = vP3Dc[i](0) * invz;
        const float y = vP3Dc[i](1) * invz;

        vP2D.push_back(Eigen::Vector2d(fx * x + cx, fy * y + cy));
    }
}

} // namespace SLAM
