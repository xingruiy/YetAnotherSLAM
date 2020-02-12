#include "Mapping.h"
#include "Matcher.h"
#include "Bundler.h"
#include "Converter.h"

namespace SLAM
{

Mapping::Mapping(const std::string &strVocFile, Map *map) : mpMap(map)
{
    ORBExtractor = new ORB_SLAM2::ORBextractor(g_ORBNFeatures, g_ORBScaleFactor, g_ORBNLevels, g_ORBIniThFAST, g_ORBMinThFAST);

    std::cout << "loading ORB vocabulary..." << std::endl;
    ORBvocabulary = new ORB_SLAM2::ORBVocabulary();
    ORBvocabulary->loadFromTextFile(strVocFile);
    std::cout << "ORB vocabulary loaded..." << std::endl;

    localKeyFrames = std::vector<KeyFrame *>();
    localMapPoints = std::vector<MapPoint *>();
}

void Mapping::Run()
{
    std::cout << "Mapping Thread Started." << std::endl;

    while (!g_bSystemKilled)
    {
        if (HasFrameToProcess())
        {
            MakeNewKeyFrame();

            LookforPointMatches();

            if (!HasFrameToProcess())
            {
                SearchInNeighbors();
            }

            if (!HasFrameToProcess())
            {
                KeyFrameCulling();
            }

            TriangulatePoints();

            CreateNewMapPoints();

            UpdateConnections();

            lastKeyFrame = NextKeyFrame;
        }
    }

    std::cout << "Mapping Thread Killed." << std::endl;
}

void Mapping::reset()
{
    localKeyFrames = std::vector<KeyFrame *>();
    localMapPoints = std::vector<MapPoint *>();
}

void Mapping::AddKeyFrameCandidate(const Frame &F)
{
    std::unique_lock<std::mutex> lock(frameMutex);
    mlFrameQueue.push_back(F);
}

bool Mapping::HasFrameToProcess()
{
    std::unique_lock<std::mutex> lock(frameMutex);
    return (!mlFrameQueue.empty());
}

void Mapping::MakeNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(frameMutex);
        NextFrame = mlFrameQueue.front();
        mlFrameQueue.pop_front();
    }

    // Create new keyframe
    NextKeyFrame = new KeyFrame(&NextFrame, mpMap, ORBExtractor);
    NextKeyFrame->ComputeBoW(ORBvocabulary);

    // Update Frame Pose
    if (lastKeyFrame != NULL)
    {
        // NextKeyFrame->mTcw = lastKeyFrame->mTcw * NextFrame.T_frame2Ref;
        // NextKeyFrame->mReferenceKeyFrame = lastKeyFrame;
    }

    // Create map points for the first frame
    if (NextKeyFrame->mnId == 0)
    {
        for (int i = 0; i < NextKeyFrame->mvKeysUn.size(); ++i)
        {
            const float d = NextKeyFrame->mvDepth[i];
            if (d > 0)
            {
                auto posWorld = NextKeyFrame->UnprojectKeyPoint(i);
                MapPoint *pMP = new MapPoint(posWorld, mpMap, NextKeyFrame, i);
                pMP->AddObservation(NextKeyFrame, i);
                pMP->UpdateNormalAndDepth();
                pMP->ComputeDistinctiveDescriptors();

                NextKeyFrame->AddMapPoint(pMP, i);
                mpMap->AddMapPoint(pMP);
            }
        }
    }

    // Insert the keyframe in the map
    mpMap->AddKeyFrame(NextKeyFrame);
}

void Mapping::LookforPointMatches()
{
    if (localMapPoints.size() == 0)
        return;

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (auto vit = localMapPoints.begin(), vend = localMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP->isBad())
            continue;

        // Project (this fills MapPoint variables for matching)
        if (NextKeyFrame->IsInFrustum(pMP, 0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0)
    {
        Matcher matcher(0.8);
        // Project points to the current keyframe
        // And search for potential corresponding points
        nToMatch = matcher.SearchByProjection(NextKeyFrame, localMapPoints, 3);
    }

    // Update covisibility based on the correspondences
    NextKeyFrame->UpdateConnections();

    std::cout << "matched map points: " << nToMatch;
}

void Mapping::KeyFrameCulling()
{
    auto vpLocalKeyFrames = NextKeyFrame->GetVectorCovisibleKeyFrames();
    for (auto vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
    {
        KeyFrame *pKF = *vit;
        if (pKF->mnId == 0)
            continue;

        const std::vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs = nObs;
        int nRedundantObservations = 0;
        int nMPs = 0;
        for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
        {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP)
            {
                if (!pMP->isBad())
                {

                    if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
                        continue;

                    nMPs++;
                    if (pMP->Observations() > thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const std::map<KeyFrame *, size_t> observations = pMP->GetObservations();
                        int nObs = 0;
                        for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                        {
                            KeyFrame *pKFi = mit->first;
                            if (pKFi == pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if (scaleLeveli <= scaleLevel + 1)
                            {
                                nObs++;
                                if (nObs >= thObs)
                                    break;
                            }
                        }
                        if (nObs >= thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if (nRedundantObservations > 0.9 * nMPs)
            pKF->SetBadFlag();
    }
}

void Mapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    const auto vpNeighKFs = NextKeyFrame->GetBestCovisibilityKeyFrames(10);
    if (vpNeighKFs.size() == 0)
        return;

    std::vector<KeyFrame *> vpTargetKFs;
    for (auto vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == NextKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = NextKeyFrame->mnId;

        // Extend to some second neighbors
        const auto vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (auto vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
        {
            KeyFrame *pKFi2 = *vit2;
            if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == NextKeyFrame->mnId || pKFi2->mnId == NextKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    Matcher matcher;
    auto vpMapPointMatches = NextKeyFrame->GetMapPointMatches();
    for (auto vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;
        matcher.Fuse(pKFi, vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    std::vector<MapPoint *> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

    for (auto vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
    {
        KeyFrame *pKFi = *vitKF;
        auto vpMapPointsKFi = pKFi->GetMapPointMatches();
        for (auto vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
        {
            MapPoint *pMP = *vitMP;
            if (!pMP)
                continue;
            if (pMP->isBad() || pMP->mnFuseCandidateForKF == NextKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = NextKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(NextKeyFrame, vpFuseCandidates);

    // Update points
    vpMapPointMatches = NextKeyFrame->GetMapPointMatches();
    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    NextKeyFrame->UpdateConnections();
}

void Mapping::TriangulatePoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    const auto vpNeighKFs = NextKeyFrame->GetBestCovisibilityKeyFrames(10);
    if (vpNeighKFs.size() == 0)
        return;

    Matcher matcher(0.6, false);

    cv::Mat Rcw1 = NextKeyFrame->GetRotation();
    cv::Mat Ow1 = NextKeyFrame->GetTranslation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat twc1 = -Rcw1 * Ow1;
    cv::Mat Twc1(3, 4, CV_32F);
    Rcw1.copyTo(Twc1.colRange(0, 3));
    twc1.copyTo(Twc1.col(3));

    const float &fx1 = NextKeyFrame->fx;
    const float &fy1 = NextKeyFrame->fy;
    const float &cx1 = NextKeyFrame->cx;
    const float &cy1 = NextKeyFrame->cy;
    const float &invfx1 = NextKeyFrame->invfx;
    const float &invfy1 = NextKeyFrame->invfy;

    const float ratioFactor = 1.5f * NextKeyFrame->mfScaleFactor;

    int nnew = 0;
    // Search matches with epipolar restriction and triangulate
    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        if (i > 0 && HasFrameToProcess())
            return;

        KeyFrame *pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetTranslation();
        cv::Mat vBaseline = Ow2 - Ow1;
        const float baseline = cv::norm(vBaseline);

        if (baseline < pKF2->mb)
            continue;

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(NextKeyFrame, pKF2);

        // Search matches that fullfil epipolar constraint
        std::vector<std::pair<size_t, size_t>> vMatchedIndices;
        matcher.SearchForTriangulation(NextKeyFrame, pKF2, F12, vMatchedIndices, false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat twc2 = -Rwc2 * Ow2;
        cv::Mat Twc2(3, 4, CV_32F);
        Rwc2.copyTo(Twc2.colRange(0, 3));
        twc2.copyTo(Twc2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for (int ikp = 0; ikp < nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = NextKeyFrame->mvKeysUn[idx1];
            const float kp1_ur = NextKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur >= 0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur >= 0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if (bStereo1)
                cosParallaxStereo1 = std::cos(2 * std::atan2(NextKeyFrame->mb / 2, NextKeyFrame->mvDepth[idx1]));
            else if (bStereo2)
                cosParallaxStereo2 = std::cos(2 * std::atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

            cosParallaxStereo = std::min(cosParallaxStereo1, cosParallaxStereo2);

            cv::Mat x3D;
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<float>(0) * Twc1.row(2) - Twc1.row(0);
                A.row(1) = xn1.at<float>(1) * Twc1.row(2) - Twc1.row(1);
                A.row(2) = xn2.at<float>(0) * Twc2.row(2) - Twc2.row(0);
                A.row(3) = xn2.at<float>(1) * Twc2.row(2) - Twc2.row(1);

                cv::Mat w, u, vt;
                cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if (x3D.at<float>(3) == 0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
            }
            else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
            {
                x3D = ORB_SLAM2::Converter::toCvMat(NextKeyFrame->UnprojectKeyPoint(idx1));
            }
            else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
            {
                x3D = ORB_SLAM2::Converter::toCvMat(pKF2->UnprojectKeyPoint(idx2));
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rwc1.row(2).dot(x3Dt) + twc1.at<float>(2);
            if (z1 <= 0)
                continue;

            float z2 = Rwc2.row(2).dot(x3Dt) + twc2.at<float>(2);
            if (z2 <= 0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = NextKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rwc1.row(0).dot(x3Dt) + twc1.at<float>(0);
            const float y1 = Rwc1.row(1).dot(x3Dt) + twc1.at<float>(1);
            const float invz1 = 1.0 / z1;

            if (!bStereo1)
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float u1_r = u1 - NextKeyFrame->mbf * invz1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rwc2.row(0).dot(x3Dt) + twc2.at<float>(0);
            const float y2 = Rwc2.row(1).dot(x3Dt) + twc2.at<float>(1);
            const float invz2 = 1.0 / z2;
            if (!bStereo2)
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float u2_r = u2 - NextKeyFrame->mbf * invz2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D - Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D - Ow2;
            float dist2 = cv::norm(normal2);

            if (dist1 == 0 || dist2 == 0)
                continue;

            const float ratioDist = dist2 / dist1;
            const float ratioOctave = NextKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint *pMP = new MapPoint(ORB_SLAM2::Converter::toVector3d(x3D), NextKeyFrame, mpMap);

            pMP->AddObservation(NextKeyFrame, idx1);
            pMP->AddObservation(pKF2, idx2);

            NextKeyFrame->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }

    std::cout << "triangulated points: " << nnew << std::endl;
}

void Mapping::CreateNewMapPoints()
{
    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(NextKeyFrame->N);
    for (int i = 0; i < NextKeyFrame->N; i++)
    {
        float z = NextKeyFrame->mvDepth[i];
        if (z > 0)
        {
            vDepthIdx.push_back(std::make_pair(z, i));
        }
    }

    int nPoints = 0;

    if (!vDepthIdx.empty())
    {
        std::sort(vDepthIdx.begin(), vDepthIdx.end());
        for (size_t j = 0; j < vDepthIdx.size(); j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;
            MapPoint *pMP = NextKeyFrame->mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
            {
                bCreateNew = true;
                NextKeyFrame->mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }

            if (bCreateNew)
            {
                auto x3D = NextKeyFrame->UnprojectKeyPoint(i);
                MapPoint *pNewMP = new MapPoint(x3D, NextKeyFrame, mpMap);
                pNewMP->AddObservation(NextKeyFrame, i);
                NextKeyFrame->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);
                NextKeyFrame->AddMapPoint(pNewMP, i);
                nPoints++;
            }
            else
            {
                nPoints++;
            }

            if (vDepthIdx[j].first > g_thDepth && nPoints > 100)
                break;
        }
    }

    std::cout << "points created in the keyframe: " << nPoints << std::endl;
}

void Mapping::UpdateConnections()
{
    // Each map point vote for the keyframes
    // in which it has been observed
    std::map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < NextKeyFrame->mvpMapPoints.size(); i++)
    {
        MapPoint *pMP = NextKeyFrame->mvpMapPoints[i];
        if (pMP && !pMP->isBad())
        {
            const std::map<KeyFrame *, size_t> observations = pMP->GetObservations();
            for (auto it = observations.begin(), itend = observations.end(); it != itend; it++)
                keyframeCounter[it->first]++;
        }
    }

    // I.e. no keyframe in the vicinity
    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    localKeyFrames.clear();
    localKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map.
    // Also check which keyframe shares most points, i.e. pKFmax
    for (auto it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (it->second > max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        localKeyFrames.push_back(it->first);
    }

    if (pKFmax)
        referenceKeyframe = pKFmax;

    // Update local map points
    // All points in the local map is included
    localMapPoints.clear();
    for (auto itKF = localKeyFrames.begin(), itEndKF = localKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        // Get map points in the keyframe
        const std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (auto itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == NextKeyFrame->mnId)
                continue;
            if (!pMP->isBad())
            {
                localMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = NextKeyFrame->mnId;
            }
        }
    }

    std::cout << "local frame: " << localKeyFrames.size() << std::endl;
    std::cout << "local points: " << localMapPoints.size() << std::endl;
}

cv::Mat Mapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation().t();
    cv::Mat t1w = -R1w * pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation().t();
    cv::Mat t2w = -R2w * pKF2->GetTranslation();

    cv::Mat R12 = R1w * R2w.t();
    cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    return K1.t().inv() * t12x * R12 * K2.inv();
}

cv::Mat Mapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2), 0, -v.at<float>(0),
            -v.at<float>(1), v.at<float>(0), 0);
}

} // namespace SLAM