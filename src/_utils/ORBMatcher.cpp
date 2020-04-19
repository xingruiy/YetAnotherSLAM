#include <DBoW2/DBoW2/FeatureVector.h>
#include "ORBMatcher.h"
#include <sophus/sim3.hpp>

namespace slam
{

const int ORBMatcher::TH_HIGH = 100;
const int ORBMatcher::TH_LOW = 50;
const int ORBMatcher::HISTO_LENGTH = 30;

ORBMatcher::ORBMatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

float ORBMatcher::RadiusByViewingCos(const float &viewCos)
{
    if (viewCos > 0.998)
        return 2.5;
    else
        return 4.0;
}

int ORBMatcher::SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th)
{
    int nmatches = 0;

    const bool bFactor = th != 1.0;

    int nMP = vpMapPoints.size();
    for (size_t iMP = 0; iMP < nMP; iMP++)
    {
        MapPoint *pMP = vpMapPoints[iMP];
        if (!pMP)
            continue;

        if (!pMP->mbTrackInView || pMP->isBad())
            continue;

        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);
        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        if (bFactor)
            r *= th;

        const auto vIndices = F.GetFeaturesInArea(pMP->mTrackProjX,
                                                  pMP->mTrackProjY,
                                                  r, //* F.mvScaleFactors[nPredictedLevel],
                                                  nPredictedLevel - 2,
                                                  nPredictedLevel);

        if (vIndices.empty())
            continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist = 256;
        int bestLevel = -1;
        int bestDist2 = 256;
        int bestLevel2 = -1;
        int bestIdx = -1;

        // Get best and second matches with near keypoints
        for (auto vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            if (F.pointsMatches[idx])
                if (F.pointsMatches[idx]->Observations() > 0)
                    continue;

            if (F.mvuRight[idx] < 0)
                continue;

            if (fabs(F.mvDepth[idx] - pMP->mTrackProjZ) > 0.05)
                continue;

            // if (F.mvuRight[idx] > 0)
            // {
            //     const float er = fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
            //     if (er > r * F.mvScaleFactors[nPredictedLevel])
            //         continue;
            // }

            const cv::Mat &d = F.mDescriptors.row(idx);
            const int dist = DescriptorDistance(MPdescriptor, d);

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx = idx;
            }
            else if (dist < bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2 = dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if (bestDist <= TH_HIGH)
        {
            if (bestLevel == bestLevel2)
                if (bestDist > mfNNratio * bestDist2)
                {
                    continue;
                }

            F.pointsMatches[bestIdx] = pMP;
            nmatches++;
        }
    }

    std::cout << "preliminary matches: " << nmatches << std::endl;
    return nmatches;
}

int ORBMatcher::SearchByProjection(Frame &F, KeyFrame *pKF, const std::set<MapPoint *> &sAlreadyFound, const float th, const int ORBdist)
{
    int nmatches = 0;

    const Eigen::Vector3d Ow = F.mTcw.translation();

    // Rotation Histogram (to check rotation consistency)
    std::vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    const std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMPs[i];

        if (pMP)
        {
            if (!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                Eigen::Vector3d x3Dw = pMP->GetWorldPos();
                Eigen::Vector3d x3Dc = F.mTcw * x3Dw;

                const float xc = x3Dc(0);
                const float yc = x3Dc(1);
                const float invzc = 1.0 / x3Dc(2);

                const float u = F.fx * xc * invzc + F.cx;
                const float v = F.fy * yc * invzc + F.cy;

                if (u < F.mnMinX || u > F.mnMaxX)
                    continue;
                if (v < F.mnMinY || v > F.mnMaxY)
                    continue;

                // Compute predicted scale level
                Eigen::Vector3d PO = x3Dw - Ow;
                float dist3D = PO.norm();

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if (dist3D < minDistance || dist3D > maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D, &F);

                // Search in a window
                const float radius = th * F.mvScaleFactors[nPredictedLevel];

                const std::vector<size_t> vIndices2 = F.GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

                if (vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for (auto vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if (F.pointsMatches[i2])
                        continue;

                    const cv::Mat &d = F.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP, d);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= ORBdist)
                {
                    F.pointsMatches[bestIdx2] = pMP;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle - F.mvKeysUn[bestIdx2].angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    F.pointsMatches[rotHist[i][j]] = NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int ORBMatcher::SearchByBoW(Frame &F, KeyFrame *pKF, std::vector<MapPoint *> &vpMapPointMatches)
{
    const std::vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = std::vector<MapPoint *>(F.N, nullptr);

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches = 0;

    std::vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while (KFit != KFend && Fit != Fend)
    {
        if (KFit->first == Fit->first)
        {
            const std::vector<unsigned int> vIndicesKF = KFit->second;
            const std::vector<unsigned int> vIndicesF = Fit->second;

            for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint *pMP = vpMapPointsKF[realIdxKF];

                if (!pMP || pMP->isBad())
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

                int bestDist1 = 256;
                int bestIdxF = -1;
                int bestDist2 = 256;

                for (size_t iF = 0; iF < vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if (vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                    const int dist = DescriptorDistance(dKF, dF);

                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdxF = realIdxF;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist1 <= TH_LOW)
                {
                    if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF] = pMP;

                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                        if (mbCheckOrientation)
                        {
                            float rot = kp.angle - F.mvKeys[bestIdxF].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }
            }

            KFit++;
            Fit++;
        }
        else if (KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]] = nullptr;
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBMatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12)
{
    const std::vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const std::vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const std::vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const std::vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = std::vector<MapPoint *>(vpMapPoints1.size(), nullptr);
    std::vector<bool> vbMatched2(vpMapPoints2.size(), false);

    std::vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint *pMP1 = vpMapPoints1[idx1];
                if (!pMP1)
                    continue;
                if (pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1 = 256;
                int bestIdx2 = -1;
                int bestDist2 = 256;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint *pMP2 = vpMapPoints2[idx2];

                    if (vbMatched2[idx2] || !pMP2)
                        continue;

                    if (pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1, d2);

                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx2 = idx2;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist1 < TH_LOW)
                {
                    if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1] = vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2] = true;

                        if (mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vpMatches12[rotHist[i][j]] = nullptr;
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBMatcher::SearchByProjection(KeyFrame *pKF, Sophus::SE3d Scw, const std::vector<MapPoint *> &vpPoints, std::vector<MapPoint *> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    Sophus::SE3d Swc = Scw.inverse();
    Eigen::Vector3d Ow = Scw.translation();

    // Set of MapPoints already found in the KeyFrame
    std::set<MapPoint *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(nullptr);

    int nmatches = 0;

    // For each Candidate MapPoint Project and Match
    for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++)
    {
        MapPoint *pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if (pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        Eigen::Vector3d p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3d p3Dc = Swc * p3Dw;

        // Depth must be positive
        if (p3Dc(2) < 0.0)
            continue;

        // Project into Image
        const float invz = 1 / p3Dc(2);
        const float x = p3Dc(0) * invz;
        const float y = p3Dc(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF->IsInImage(u, v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3d PO = p3Dw - Ow;
        const float dist = PO.norm();

        if (dist < minDistance || dist > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3d Pn = pMP->GetNormal();

        if (PO.dot(Pn) < 0.5 * dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist, pKF);

        // Search in a radius
        const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

        const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for (auto vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;
            if (vpMatched[idx])
                continue;

            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_LOW)
        {
            vpMatched[bestIdx] = pMP;
            nmatches++;
        }
    }

    return nmatches;
}

int ORBMatcher::Fuse(KeyFrame *pKF, const std::vector<MapPoint *> &vpMapPoints, const float th)
{
    Sophus::SE3d Twc = pKF->GetPoseInverse();
    Eigen::Vector3d Ow = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    int nFused = 0;

    const int nMPs = vpMapPoints.size();

    for (int i = 0; i < nMPs; i++)
    {
        MapPoint *pMP = vpMapPoints[i];

        if (!pMP)
            continue;

        if (pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        Eigen::Vector3d p3Dw = pMP->GetWorldPos();
        Eigen::Vector3d p3Dc = Twc * p3Dw;

        // Depth must be positive
        if (p3Dc(2) < 0.0f)
            continue;

        const float invz = 1 / p3Dc(2);
        const float x = p3Dc(0) * invz;
        const float y = p3Dc(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF->IsInImage(u, v))
            continue;

        const float ur = u - bf * invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3d PO = p3Dw - Ow;
        const float dist3D = PO.norm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3d Pn = pMP->GetNormal();

        if (PO.dot(Pn) < 0.5 * dist3D)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

        // Search in a radius
        const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

        const auto vIndices = pKF->GetFeaturesInArea(u, v, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        // Eigen::Vector3d PointNormal = pMP->mPointNormal;
        for (auto vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;
            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];
            const int &kpLevel = kp.octave;
            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            if (pKF->mvuRight[idx] >= 0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u - kpx;
                const float ey = v - kpy;
                const float er = ur - kpr;
                const float e2 = ex * ex + ey * ey + er * er;

                if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u - kpx;
                const float ey = v - kpy;
                const float e2 = ex * ex + ey * ey;

                if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
                    continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);
            const int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if (bestDist <= TH_LOW)
        {
            MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
            if (pMPinKF)
            {
                if (!pMPinKF->isBad())
                {
                    if (pMPinKF->Observations() > pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
    std::cout << nFused << " Points fused together." << std::endl;
}

int ORBMatcher::Fuse(KeyFrame *pKF, Sophus::SE3d Tcw, const std::vector<MapPoint *> &vpPoints, float th, std::vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Tcw
    Sophus::SE3d Twc = Tcw.inverse();
    Eigen::Vector3d Ow = Tcw.translation();

    // Set of MapPoints already found in the KeyFrame
    const std::set<MapPoint *> spAlreadyFound = pKF->GetMapPoints();

    int nFused = 0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for (int iMP = 0; iMP < nPoints; iMP++)
    {
        MapPoint *pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if (pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        Eigen::Vector3d p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3d p3Dc = Twc * p3Dw;

        // Depth must be positive
        if (p3Dc(2) < 0.0f)
            continue;

        // Project into Image
        const float invz = 1.0 / p3Dc(2);
        const float x = p3Dc(0) * invz;
        const float y = p3Dc(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF->IsInImage(u, v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3d PO = p3Dw - Ow;
        const float dist3D = PO.norm();

        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3d Pn = pMP->GetNormal();

        if (PO.dot(Pn) < 0.5 * dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

        // Search in a radius
        const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

        const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for (auto vit = vIndices.begin(); vit != vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if (bestDist <= TH_LOW)
        {
            MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
            if (pMPinKF)
            {
                if (!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }

            nFused++;
        }
    }

    return nFused;
}

int ORBMatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       std::vector<std::pair<size_t, size_t>> &vMatchedPairs,
                                       const bool bOnlyStereo)
{
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    Eigen::Vector3d Cw = pKF1->GetTranslation();
    Eigen::Vector3d C2 = pKF2->GetPoseInverse() * Cw;

    const float invz = 1.0f / C2(2);
    const float ex = pKF2->fx * C2(0) * invz + pKF2->cx;
    const float ey = pKF2->fy * C2(1) * invz + pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches = 0;
    std::vector<bool> vbMatched2(pKF2->N, false);
    std::vector<int> vMatches12(pKF1->N, -1);

    std::vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint *pMP1 = pKF1->GetMapPoint(idx1);

                // If there is already a MapPoint skip
                if (pMP1)
                    continue;

                const bool bStereo1 = pKF1->mvuRight[idx1] >= 0;

                if (bOnlyStereo)
                    if (!bStereo1)
                        continue;

                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];

                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                int bestDist = TH_LOW;
                int bestIdx2 = -1;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    MapPoint *pMP2 = pKF2->GetMapPoint(idx2);

                    // If we have already matched or there is a MapPoint skip
                    if (vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2] >= 0;

                    if (bOnlyStereo)
                        if (!bStereo2)
                            continue;

                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                    const int dist = DescriptorDistance(d1, d2);

                    if (dist > TH_LOW || dist > bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if (!bStereo1 && !bStereo2)
                    {
                        const float distex = ex - kp2.pt.x;
                        const float distey = ey - kp2.pt.y;
                        if (distex * distex + distey * distey < 100 * pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

                    if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                if (bestIdx2 >= 0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1] = bestIdx2;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        float rot = kp1.angle - kp2.angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vMatches12[rotHist[i][j]] = -1;
                nmatches--;
            }
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (vMatches12[i] < 0)
            continue;
        vMatchedPairs.push_back(std::make_pair(i, vMatches12[i]));
    }

    return nmatches;
}

int ORBMatcher::SearchBySE3(Frame &F, KeyFrame *pKF, std::vector<MapPoint *> &vpMatches12, const Sophus::SE3d &Tcp, const float th)
{
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // auto Twc1 = F.mTcw.inverse();
    auto Twc2 = pKF->GetPoseInverse();

    // Transformation between cameras
    Sophus::SE3d Tpc = Tcp.inverse();

    // const std::vector<MapPoint *> vpMapPoints1 = F.pointsMatches;
    // const int N1 = vpMapPoints1.size();
    const int N1 = F.N;

    const std::vector<MapPoint *> vpMapPoints2 = pKF->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    std::vector<bool> vbAlreadyMatched1(N1, false);
    std::vector<bool> vbAlreadyMatched2(N2, false);
    int nAlreadyMatched = 0;
    for (int i = 0; i < N1; i++)
    {
        MapPoint *pMP = vpMatches12[i];
        if (pMP)
        {
            vbAlreadyMatched1[i] = true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF);
            if (idx2 >= 0 && idx2 < N2)
                vbAlreadyMatched2[idx2] = true;
            nAlreadyMatched++;
        }
    }

    std::cout << "num already matched " << nAlreadyMatched << std::endl;

    std::vector<int> vnMatch1(N1, -1);
    std::vector<int> vnMatch2(N2, -1);

    // Transform from KF1 to KF2 and search
    for (int i1 = 0; i1 < N1; i1++)
    {
        // MapPoint *pMP = vpMapPoints1[i1];
        if (F.mvuRight[i1] <= 0)
            continue;

        if (vbAlreadyMatched1[i1])
            continue;

        Eigen::Vector3d p3Dc1 = F.mvRelocPoints[i1];
        Eigen::Vector3d p3Dc2 = Tcp * p3Dc1;

        // Depth must be positive
        if (p3Dc2(2) < 0.0)
            continue;

        const float invz = 1.0 / p3Dc2(2);
        const float x = p3Dc2(0) * invz;
        const float y = p3Dc2(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF->IsInImage(u, v))
            continue;

        // const float maxDistance = pMP->GetMaxDistanceInvariance();
        // const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc2.norm();

        // // Depth must be inside the scale invariance region
        // if (dist3D < minDistance || dist3D > maxDistance)
        //     continue;

        // Compute predicted octave
        // const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

        // Search in a radius
        // const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

        const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, 15);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = F.mDescriptors.row(i1); // pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for (auto vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            // if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
            //     continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_HIGH)
        {
            vnMatch1[i1] = bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for (int i2 = 0; i2 < N2; i2++)
    {
        MapPoint *pMP = vpMapPoints2[i2];

        if (!pMP || vbAlreadyMatched2[i2])
            continue;

        if (pMP->isBad())
            continue;

        Eigen::Vector3d p3Dw = pMP->GetWorldPos();
        Eigen::Vector3d p3Dc2 = Twc2 * p3Dw;
        Eigen::Vector3d p3Dc1 = Tpc * p3Dc2;

        // Depth must be positive
        if (p3Dc1(2) < 0.0)
            continue;

        const float invz = 1.0 / p3Dc1(2);
        const float x = p3Dc1(0) * invz;
        const float y = p3Dc1(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF->IsInImage(u, v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc1.norm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D, &F);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th * F.mvScaleFactors[nPredictedLevel];

        const std::vector<size_t> vIndices = F.GetFeaturesInArea(u, v, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for (auto vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
                continue;

            const cv::Mat &dKF = F.mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_HIGH)
        {
            vnMatch2[i2] = bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for (int i1 = 0; i1 < N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if (idx2 >= 0)
        {
            int idx1 = vnMatch2[idx2];
            if (idx1 == i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

int ORBMatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::SE3d &S12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    auto Twc1 = pKF1->GetPoseInverse();
    auto Twc2 = pKF2->GetPoseInverse();

    //Transformation between cameras
    Sophus::SE3d S21 = S12.inverse();

    const std::vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const std::vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    std::vector<bool> vbAlreadyMatched1(N1, false);
    std::vector<bool> vbAlreadyMatched2(N2, false);

    for (int i = 0; i < N1; i++)
    {
        MapPoint *pMP = vpMatches12[i];
        if (pMP)
        {
            vbAlreadyMatched1[i] = true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if (idx2 >= 0 && idx2 < N2)
                vbAlreadyMatched2[idx2] = true;
        }
    }

    std::vector<int> vnMatch1(N1, -1);
    std::vector<int> vnMatch2(N2, -1);

    // Transform from KF1 to KF2 and search
    for (int i1 = 0; i1 < N1; i1++)
    {
        MapPoint *pMP = vpMapPoints1[i1];

        if (!pMP || vbAlreadyMatched1[i1])
            continue;

        if (pMP->isBad())
            continue;

        Eigen::Vector3d p3Dw = pMP->GetWorldPos();
        Eigen::Vector3d p3Dc1 = Twc1 * p3Dw;
        Eigen::Vector3d p3Dc2 = S12 * p3Dc1;

        // Depth must be positive
        if (p3Dc2(2) < 0.0)
            continue;

        const float invz = 1.0 / p3Dc2(2);
        const float x = p3Dc2(0) * invz;
        const float y = p3Dc2(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF2->IsInImage(u, v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc2.norm();

        // Depth must be inside the scale invariance region
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

        // Search in a radius
        const float radius = th * pKF2->mvScaleFactors[nPredictedLevel];

        const std::vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for (auto vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_HIGH)
        {
            vnMatch1[i1] = bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for (int i2 = 0; i2 < N2; i2++)
    {
        MapPoint *pMP = vpMapPoints2[i2];

        if (!pMP || vbAlreadyMatched2[i2])
            continue;

        if (pMP->isBad())
            continue;

        Eigen::Vector3d p3Dw = pMP->GetWorldPos();
        Eigen::Vector3d p3Dc2 = Twc2 * p3Dw;
        Eigen::Vector3d p3Dc1 = S21 * p3Dc2;

        // Depth must be positive
        if (p3Dc1(2) < 0.0)
            continue;

        const float invz = 1.0 / p3Dc1(2);
        const float x = p3Dc1(0) * invz;
        const float y = p3Dc1(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF1->IsInImage(u, v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc1.norm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th * pKF1->mvScaleFactors[nPredictedLevel];

        const std::vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for (auto vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_HIGH)
        {
            vnMatch2[i2] = bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for (int i1 = 0; i1 < N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if (idx2 >= 0)
        {
            int idx1 = vnMatch2[idx2];
            if (idx1 == i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

void ORBMatcher::ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++)
    {
        const int s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
}

bool ORBMatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
    const float b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
    const float c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

    const float num = a * kp2.pt.x + b * kp2.pt.y + c;

    const float den = a * a + b * b;

    if (den == 0)
        return false;

    const float dsqr = num * num / den;

    return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} // namespace slam
