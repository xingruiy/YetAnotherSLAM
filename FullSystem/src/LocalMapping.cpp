#include "LocalMapping.h"
#include "ORBMatcher.h"
#include "Optimizer.h"
#include "Converter.h"

namespace SLAM
{

LocalMapping::LocalMapping(ORBVocabulary *pVoc, Map *pMap)
    : ORBvocabulary(pVoc), mpLastKeyFrame(nullptr),
      mpMap(pMap), mbAbortBA(false), mbStopped(false),
      mbStopRequested(false), mbNotStop(false)
{
}

void LocalMapping::Run()
{
    while (!g_bSystemKilled)
    {
        if (CheckNewKeyFrames())
        {
            ProcessNewKeyFrame();
            std::cout << "Process New KeyFrame: " << mpCurrentKeyFrame->mnId << std::endl;

            int nMatches = MatchLocalPoints();
            std::cout << "Match Local Points: " << nMatches << std::endl;

            if (nMatches > 0)
            {
                std::cout << "Do Pose Optimization" << std::endl;
                Optimizer::PoseOptimization(mpCurrentKeyFrame);
            }

            std::cout << "Update New MapPoints" << std::endl;
            UpdateKeyFrame();
            CreateNewMapPoints(); // Create new points from depth observations

            if (!CheckNewKeyFrames() && mpCurrentKeyFrame->mnId != 0)
            {
                std::cout << "Do Local Bundle Adjustment" << std::endl;
                SearchInNeighbors();
                bool bStopFlag;
                Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &bStopFlag, mpMap);
            }

            std::cout << "Map Point Culling" << std::endl;
            UpdateLocalMap();
            MapPointCulling();

            std::cout << "KeyFrame Culling" << std::endl;
            if (!CheckNewKeyFrames())
                KeyFrameCulling();

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

            // Update reference keyframe
            mpLastKeyFrame = mpCurrentKeyFrame;
        }
        else if (Stop())
        {
            // Safe area to stop
            while (isStopped())
            {
                usleep(3000);
            }
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

void LocalMapping::AddKeyFrameCandidate(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
}

bool LocalMapping::CheckNewKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    if (mpCurrentKeyFrame == NULL)
        return;

    // Process new keyframe
    mpCurrentKeyFrame->ComputeBoW(ORBvocabulary);

    // Update Frame Pose
    if (mpCurrentKeyFrame->mnId != 0)
    {
        mpCurrentKeyFrame->mTcw = mpLastKeyFrame->mTcw * mpCurrentKeyFrame->mRelativePose;
        mpCurrentKeyFrame->mReferenceKeyFrame = mpLastKeyFrame;
    }
    // Create map points for the first frame
    else
    {
        size_t nNewPoints = 0;
        for (int i = 0; i < mpCurrentKeyFrame->mvKeysUn.size(); ++i)
        {
            const float d = mpCurrentKeyFrame->mvDepth[i];
            if (d > 0)
            {
                auto posWorld = mpCurrentKeyFrame->UnprojectKeyPoint(i);
                MapPoint *pMP = new MapPoint(posWorld, mpMap, mpCurrentKeyFrame, i);
                pMP->AddObservation(mpCurrentKeyFrame, i);
                pMP->UpdateNormalAndDepth();
                pMP->ComputeDistinctiveDescriptors();

                mpCurrentKeyFrame->AddMapPoint(pMP, i);
                mpMap->AddMapPoint(pMP);
                mvpLocalMapPoints.push_back(pMP);
                nNewPoints++;
            }
        }

        mvpLocalKeyFrames.push_back(mpCurrentKeyFrame);
        std::cout << "New Map Created With Points: " << nNewPoints << std::endl;
    }

    // Insert the keyframe in the map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);

    if (g_bEnableViewer)
        mpViewer->setKeyFrameImage(mpCurrentKeyFrame->mImg, mpCurrentKeyFrame->mvKeys);
}

int LocalMapping::MatchLocalPoints()
{
    if (mvpLocalKeyFrames.size() == 0)
        return 0;

    if (mpCurrentKeyFrame->mnId == 0)
        return 0;

    int nToMatch = 0;
    // Project points in frame and check its visibility
    for (auto vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (!pMP || pMP->isBad())
            continue;

        // Project (this fills MapPoint variables for matching)
        if (mpCurrentKeyFrame->IsInFrustum(pMP, 0.5))
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
        nToMatch = matcher.SearchByProjection(mpCurrentKeyFrame, mvpLocalMapPoints, 1);
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
    auto vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
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
            pKF->SetBadFlag();
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    const auto vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(10);
    std::cout << "Covisibility Graph Size: " << vpNeighKFs.size() << std::endl;
    if (vpNeighKFs.size() == 0)
        return;

    std::vector<KeyFrame *> vpTargetKFs;
    for (auto vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const auto vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (auto vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
        {
            KeyFrame *pKFi2 = *vit2;
            if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBMatcher matcher;
    auto vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
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
            if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
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
    mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::CreateNewMapPoints()
{
    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mpCurrentKeyFrame->N);
    for (int i = 0; i < mpCurrentKeyFrame->N; i++)
    {
        float z = mpCurrentKeyFrame->mvDepth[i];
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
            MapPoint *pMP = mpCurrentKeyFrame->mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
            {
                bCreateNew = true;
                mpCurrentKeyFrame->mvpMapPoints[i] = NULL;
            }

            if (bCreateNew)
            {
                Eigen::Vector3d x3D;
                if (mpCurrentKeyFrame->UnprojectKeyPoint(x3D, i))
                {
                    // mpCurrentKeyFrame->mvbOutlier[i] = false;
                    MapPoint *pNewMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);
                    pNewMP->AddObservation(mpCurrentKeyFrame, i);
                    mpCurrentKeyFrame->AddMapPoint(pNewMP, i);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();

                    mpMap->AddMapPoint(pNewMP);
                    mpCurrentKeyFrame->AddMapPoint(pNewMP, i);

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
}

void LocalMapping::UpdateLocalMap()
{
    if (mpCurrentKeyFrame->mnId == 0)
    {
        mvpLocalKeyFrames.clear();
        mvpLocalMapPoints.clear();

        auto sMapPoints = mpCurrentKeyFrame->GetMapPoints();
        mvpLocalMapPoints = std::vector<MapPoint *>(sMapPoints.begin(), sMapPoints.end());
        mvpLocalKeyFrames.push_back(mpCurrentKeyFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        return;
    }

    // Each map point vote for the keyframes
    // in which it has been observed
    std::map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < mpCurrentKeyFrame->mvpMapPoints.size(); i++)
    {
        MapPoint *pMP = mpCurrentKeyFrame->mvpMapPoints[i];
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
        mpReferenceKeyframe = pKFmax;

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
            if (pMP->mnTrackReferenceForFrame == mpCurrentKeyFrame->mnId)
                continue;
            if (!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mpCurrentKeyFrame->mnId;
            }
        }
    }

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    std::cout << "Local Map Size:  " << mvpLocalKeyFrames.size() << std::endl;
    std::cout << "Local Point Size:  " << mvpLocalMapPoints.size() << std::endl;
}

void LocalMapping::UpdateKeyFrame()
{
    size_t nOutliers = 0;
    // Update MapPoints Statistics
    for (int i = 0; i < mpCurrentKeyFrame->N; i++)
    {
        if (mpCurrentKeyFrame->mvpMapPoints[i])
        {
            if (!mpCurrentKeyFrame->mvbOutlier[i])
            {
                mpCurrentKeyFrame->mvpMapPoints[i]->IncreaseFound();
            }
            else
            {
                nOutliers++;
                mpCurrentKeyFrame->mvbOutlier[i] = false;
                mpCurrentKeyFrame->mvpMapPoints[i] = NULL;
            }
        }
    }

    const auto vpMPs = mpCurrentKeyFrame->GetMapPointMatches();
    for (int i = 0; i < vpMPs.size(); ++i)
    {
        MapPoint *pMP = vpMPs[i];
        if (!pMP || pMP->isBad())
            continue;

        pMP->AddObservation(mpCurrentKeyFrame, i);
        pMP->UpdateNormalAndDepth();
        pMP->ComputeDistinctiveDescriptors();
    }

    // Update covisibility based on the correspondences
    mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::RequestStop()
{
    {
        std::unique_lock<std::mutex> lock(mMutexStop);
        mbStopRequested = true;
    }

    {
        std::unique_lock<std::mutex> lock2(mMutexNewKFs);
        mbAbortBA = true;
    }
}

void LocalMapping::RequestReset()
{
}

bool LocalMapping::Stop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        std::cout << "Local Mapping STOP" << std::endl;
        return true;
    }

    return false;
}

void LocalMapping::Release()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    std::unique_lock<std::mutex> lock2(mMutexFinish);
    if (mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for (auto lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    std::cout << "Local Mapping RELEASE" << std::endl;
}

bool LocalMapping::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::isFinished()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

} // namespace SLAM