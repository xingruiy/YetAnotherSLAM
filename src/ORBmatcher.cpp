#include "ORBmatcher.h"

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
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

int ORBmatcher::SearchByProjection(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    int nmatches = 0;
    const bool bFactor = th != 1.0;

    for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
    {
        MapPoint *pMP = vpMapPoints[iMP];
        if (!pMP->mbTrackInView)
            continue;

        if (pMP->isBad())
            continue;

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        //     if (bFactor)
        //         r *= th;

        //     const vector<size_t> vIndices =
        //         F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r * F.mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);

        //     if (vIndices.empty())
        //         continue;

        //     const cv::Mat MPdescriptor = pMP->GetDescriptor();

        //     int bestDist = 256;
        //     int bestLevel = -1;
        //     int bestDist2 = 256;
        //     int bestLevel2 = -1;
        //     int bestIdx = -1;

        //     // Get best and second matches with near keypoints
        //     for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        //     {
        //         const size_t idx = *vit;

        //         if (F.mvpMapPoints[idx])
        //             if (F.mvpMapPoints[idx]->Observations() > 0)
        //                 continue;

        //         if (F.mvuRight[idx] > 0)
        //         {
        //             const float er = fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
        //             if (er > r * F.mvScaleFactors[nPredictedLevel])
        //                 continue;
        //         }

        //         const cv::Mat &d = F.mDescriptors.row(idx);

        //         const int dist = DescriptorDistance(MPdescriptor, d);

        //         if (dist < bestDist)
        //         {
        //             bestDist2 = bestDist;
        //             bestDist = dist;
        //             bestLevel2 = bestLevel;
        //             bestLevel = F.mvKeysUn[idx].octave;
        //             bestIdx = idx;
        //         }
        //         else if (dist < bestDist2)
        //         {
        //             bestLevel2 = F.mvKeysUn[idx].octave;
        //             bestDist2 = dist;
        //         }
        //     }

        //     // Apply ratio to second match (only if best and second are in the same scale level)
        //     if (bestDist <= TH_HIGH)
        //     {
        //         if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
        //             continue;

        //         F.mvpMapPoints[bestIdx] = pMP;
        //         nmatches++;
        //     }
    }

    return nmatches;
}
