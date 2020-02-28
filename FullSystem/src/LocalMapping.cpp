#include "LocalMapping.h"
#include "ORBMatcher.h"
#include "Bundler.h"
#include "Converter.h"

namespace SLAM
{

LocalMapping::LocalMapping(ORB_SLAM2::ORBVocabulary *pVoc, Map *pMap)
    : mpMap(pMap), ORBvocabulary(pVoc), mLastKeyFrame(NULL)
{
    mpExtractor = new ORBextractor(g_ORBNFeatures, g_ORBScaleFactor, g_ORBNLevels, g_ORBIniThFAST, g_ORBMinThFAST);
    mvpLocalKeyFrames = std::vector<KeyFrame *>();
    mvpLocalMapPoints = std::vector<MapPoint *>();
}

void LocalMapping::Run()
{
    std::cout << "LocalMapping Thread Started." << std::endl;

    while (!g_bSystemKilled)
    {
        if (HasFrameToProcess())
        {
            CreateNewKeyFrame();
            int nMatches = MatchLocalPoints();
            std::cout << "num local points matched: " << nMatches << std::endl;

            if (nMatches > 0)
            {
                Bundler::PoseOptimization(NextKeyFrame);
            }

            UpdateKeyFrame();
            CreateNewMapPoints(); // Create new points from depth observations

            if (!HasFrameToProcess())
            {
                SearchInNeighbors();
                bool bStopFlag;
                Bundler::LocalBundleAdjustment(NextKeyFrame, &bStopFlag, mpMap);
            }

            UpdateLocalMap();
            MapPointCulling();

            if (!HasFrameToProcess())
                KeyFrameCulling();

            mpLoopCloser->InsertKeyFrame(NextKeyFrame);
            // Update reference keyframe
            mLastKeyFrame = NextKeyFrame;
        }
    }
}

void LocalMapping::setLoopCloser(LoopClosing *pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::setViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

void LocalMapping::reset()
{
    mvpLocalKeyFrames = std::vector<KeyFrame *>();
    mvpLocalMapPoints = std::vector<MapPoint *>();
}

void LocalMapping::AddKeyFrameCandidate(const Frame &F)
{
    std::unique_lock<std::mutex> lock(frameMutex);
    mlFrameQueue.push_back(F);
}

bool LocalMapping::HasFrameToProcess()
{
    std::unique_lock<std::mutex> lock(frameMutex);
    return (!mlFrameQueue.empty());
}

void LocalMapping::CreateNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(frameMutex);
        mCurrentFrame = mlFrameQueue.front();
        mlFrameQueue.pop_front();
    }

    // Create new keyframe
    NextKeyFrame = new KeyFrame(&mCurrentFrame, mpMap, mpExtractor);
    NextKeyFrame->ComputeBoW(ORBvocabulary);

    std::cout << "============================\n"
              << "processing keyframe: "
              << NextKeyFrame->mnId << std::endl;

    // Update Frame Pose
    if (mLastKeyFrame != NULL)
    {
        NextKeyFrame->mTcw = mLastKeyFrame->mTcw * mCurrentFrame.mRelativePose;
        NextKeyFrame->mReferenceKeyFrame = mLastKeyFrame;
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
                pMP->UpdateDepthAndViewingDir();
                pMP->ComputeDistinctiveDescriptors();

                NextKeyFrame->AddMapPoint(pMP, i);
                mpMap->AddMapPoint(pMP);
                mvpLocalMapPoints.push_back(pMP);
            }
        }
    }

    // Insert the keyframe in the map
    mpMap->AddKeyFrame(NextKeyFrame);

    if (g_bEnableViewer)
        mpViewer->setKeyFrameImage(mCurrentFrame.mImGray, NextKeyFrame->mvKeys);
}

int LocalMapping::MatchLocalPoints()
{
    if (mvpLocalKeyFrames.size() == 0)
        return 0;

    int nToMatch = 0;
    // Project points in frame and check its visibility
    for (auto vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (!pMP || pMP->isBad())
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
        ORBMatcher matcher(0.8);
        // Project points to the current keyframe
        // And search for potential corresponding points
        nToMatch = matcher.SearchByProjection(NextKeyFrame, mvpLocalMapPoints, 1);
    }

    return nToMatch;
}

void LocalMapping::MapPointCulling()
{
    for (auto vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; ++vit)
    {
        MapPoint *pMP = (*vit);
        float ratio = pMP->GetFoundRatio();
        if (ratio < 0.25f && pMP->Observations() < 3)
        {
            pMP->SetBadFlag();
        }
    }
}

void LocalMapping::KeyFrameCulling()
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
            if (!pMP || pMP->isBad())
                continue;

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

        if (nRedundantObservations > 0.9 * nMPs)
        {
            pKF->SetBadFlag();
            std::cout << "keyframe " << pKF->mnId << " is flaged redundant" << std::endl;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    const auto vpNeighKFs = NextKeyFrame->GetBestCovisibilityKeyFrames(10);
    if (vpNeighKFs.size() == 0)
        return;
    std::cout << "Covisible Graph Size: " << vpNeighKFs.size() << std::endl;
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
    ORBMatcher matcher;
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
                pMP->UpdateDepthAndViewingDir();
            }
        }
    }

    // Update connections in covisibility graph
    NextKeyFrame->UpdateConnections();
}

void LocalMapping::CreateNewMapPoints()
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
    int nCreated = 0;
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
                NextKeyFrame->mvpMapPoints[i] = NULL;
            }

            if (bCreateNew)
            {
                Eigen::Vector3d x3D;
                if (NextKeyFrame->UnprojectKeyPoint(x3D, i))
                {
                    // NextKeyFrame->mvbOutlier[i] = false;
                    MapPoint *pNewMP = new MapPoint(x3D, NextKeyFrame, mpMap);
                    pNewMP->AddObservation(NextKeyFrame, i);
                    NextKeyFrame->AddMapPoint(pNewMP, i);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateDepthAndViewingDir();

                    mpMap->AddMapPoint(pNewMP);
                    NextKeyFrame->AddMapPoint(pNewMP, i);

                    nPoints++;
                    nCreated++;
                }
            }
            else
                nPoints++;

            if ((vDepthIdx[j].first > g_thDepth && nPoints > 100) || nPoints >= 500)
                break;
        }
    }

    std::cout << "points created in the keyframe: " << nCreated << std::endl;
}

void LocalMapping::UpdateLocalMap()
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

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

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

        mvpLocalKeyFrames.push_back(it->first);
    }

    if (pKFmax)
        referenceKeyframe = pKFmax;

    // Update local map points
    // All points in the local map is included
    mvpLocalMapPoints.clear();
    for (auto itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        // Get map points in the keyframe
        const auto vpMPs = pKF->GetMapPointMatches();
        for (auto itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == NextKeyFrame->mnId)
                continue;
            if (!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = NextKeyFrame->mnId;
            }
        }
    }

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    std::cout << "local frame: " << mvpLocalKeyFrames.size() << std::endl;
    std::cout << "local points: " << mvpLocalMapPoints.size() << std::endl;
}

void LocalMapping::UpdateKeyFrame()
{
    size_t nOutliers = 0;
    // Update MapPoints Statistics
    for (int i = 0; i < NextKeyFrame->N; i++)
    {
        if (NextKeyFrame->mvpMapPoints[i])
        {
            if (!NextKeyFrame->mvbOutlier[i])
            {
                NextKeyFrame->mvpMapPoints[i]->IncreaseFound();
            }
            else
            {
                nOutliers++;
                NextKeyFrame->mvbOutlier[i] = false;
                NextKeyFrame->mvpMapPoints[i] = NULL;
            }
        }
    }

    const auto vpMPs = NextKeyFrame->GetMapPointMatches();
    for (int i = 0; i < vpMPs.size(); ++i)
    {
        MapPoint *pMP = vpMPs[i];
        if (!pMP || pMP->isBad())
            continue;

        pMP->AddObservation(NextKeyFrame, i);
        pMP->UpdateDepthAndViewingDir();
        pMP->ComputeDistinctiveDescriptors();
    }

    // Update covisibility based on the correspondences
    NextKeyFrame->UpdateConnections();

    std::cout << nOutliers << " outliers found in keyframe" << std::endl;
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
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

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2), 0, -v.at<float>(0),
            -v.at<float>(1), v.at<float>(0), 0);
}

} // namespace SLAM