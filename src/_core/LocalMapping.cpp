#include "LocalMapping.h"
#include "ORBMatcher.h"
#include "Optimizer.h"
#include "Map.h"

namespace slam
{

LocalMapping::LocalMapping(ORBVocabulary *pVoc, Map *mpMap)
    : mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(mpMap),
      mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false),
      mbAcceptKeyFrames(true)
{
}

void LocalMapping::Run()
{
    mbFinished = false;

    // {
    //     std::unique_lock<std::mutex> lock(mMutexAccept);
    //     mbAcceptKeyFrames = false;
    // }

    if (CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        MapPointCulling();
        CreateNewMapPoints();

        if (!CheckNewKeyFrames())
        {
            SearchInNeighbors();
        }

        mbAbortBA = false;

        if (!CheckNewKeyFrames() && !stopRequested())
        {
            // Local BA
            if (mpMap->KeyFramesInMap() > 2)
                Optimizer::LocalBundleAdjustment(currKF, &mbAbortBA, mpMap);

            // Check redundant local Keyframes
            KeyFrameCulling();
        }

        mpLoopCloser->InsertKeyFrame(currKF);
    }
    else if (Stop())
    {
        // Safe area to stop
        while (isStopped() && !CheckFinish())
        {
            usleep(3000);
        }
        if (CheckFinish())
            return;
    }

    ResetIfRequested();

    // Tracking will see that Local Mapping is busy
    // SetAcceptKeyFrames(true);

    if (CheckFinish())
        return;

    usleep(3000);
}

bool LocalMapping::AcceptKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexAccept);
    mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexStop);

    if (flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
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
        currKF = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    currKF->ComputeBoW();
    const auto vpMapPointMatches = currKF->GetMapPointMatches();

    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (!pMP->isBad())
            {
                if (!pMP->IsInKeyFrame(currKF))
                {
                    pMP->AddObservation(currKF, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    currKF->UpdateConnections();
    mpMap->AddKeyFrame(currKF);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    std::list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
    const int nCurrentKFid = currKF->mnId;

    const int cnThObs = 3;
    while (lit != mlpRecentAddedMapPoints.end())
    {
        MapPoint *pMP = *lit;
        if (pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::KeyFrameCulling()
{
    auto vpLocalKeyFrames = currKF->GetVectorCovisibleKeyFrames();
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
    const auto vpNeighKFs = currKF->GetBestCovisibilityKeyFrames(10);

    if (vpNeighKFs.size() == 0)
        return;

    std::vector<KeyFrame *> vpTargetKFs;
    for (auto vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == currKF->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = currKF->mnId;

        // Extend to some second neighbors
        const auto vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (auto vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
        {
            KeyFrame *pKFi2 = *vit2;
            if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == currKF->mnId || pKFi2->mnId == currKF->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBMatcher matcher;
    auto vpMapPointMatches = currKF->GetMapPointMatches();
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
            if (pMP->isBad() || pMP->mnFuseCandidateForKF == currKF->mnId)
                continue;
            pMP->mnFuseCandidateForKF = currKF->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(currKF, vpFuseCandidates);

    // Update points
    vpMapPointMatches = currKF->GetMapPointMatches();
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
    currKF->UpdateConnections();
}

void LocalMapping::CreateNewMapPoints()
{
}

void LocalMapping::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    std::unique_lock<std::mutex> lock2(mMutexNewKFs);
    mbStopRequested = true;
    mbAbortBA = true;
}

void LocalMapping::RequestReset()
{
    {
        std::unique_lock<std::mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while (1)
    {
        {
            std::unique_lock<std::mutex> lock(mMutexReset);
            if (!mbResetRequested)
                break;
        }

        usleep(1000);
    }
}

void LocalMapping::ResetIfRequested()
{
    std::unique_lock<std::mutex> lock(mMutexReset);
    if (mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested = false;
    }
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
    // for (auto lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
    //     delete *lit;
    // mlNewKeyFrames.clear();

    std::cout << "Local Mapping RELEASE" << std::endl;
}

bool LocalMapping::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool LocalMapping::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

bool LocalMapping::isFinished()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

} // namespace slam