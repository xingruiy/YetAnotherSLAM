#include "LocalMapping.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

LocalMapping::LocalMapping(Map *pMap)
    : mpMap(pMap), mbShouldQuit(false), mpReferenceKF(NULL)
{
}

void LocalMapping::Spin()
{
    while (!mbShouldQuit)
    {
        if (CheckNewKeyFrames())
        {
            ProcessNewKeyFrame();

            UpdateLocalMap();

            MatchLocalPoints();

            if (!CheckNewKeyFrames())
            {
                SearchInNeighbors();
            }

            mbAbortBA = false;

            if (!CheckNewKeyFrames())
            {
                // Local BA
                if (mpMap->KeyFramesInMap() > 2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);

                // Check redundant local Keyframes
                KeyFrameCulling();
            }

            CreateNewMapPoints();
        }
    }
}

void LocalMapping::KeyFrameCulling()
{
}

void LocalMapping::SearchInNeighbors()
{
}

void LocalMapping::CreateNewMapPoints()
{
    // We sort points by the measured depth by the RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mpCurrentKeyFrame->N);
    for (int i = 0; i < mpCurrentKeyFrame->N; i++)
    {
        float z = mpCurrentKeyFrame->mvDepth[i];
        if (z > 0)
        {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    if (!vDepthIdx.empty())
    {
        sort(vDepthIdx.begin(), vDepthIdx.end());

        int nPoints = 0;
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
                mpCurrentKeyFrame->mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }

            if (bCreateNew)
            {
                Eigen::Vector3d x3D = mpCurrentKeyFrame->UnprojectKeyPoint(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, mpCurrentKeyFrame, i);
                pNewMP->AddObservation(mpCurrentKeyFrame, i);
                mpCurrentKeyFrame->AddMapPoint(pNewMP, i);
                mpMap->AddMapPoint(pNewMP);
                nPoints++;
            }
            else
            {
                nPoints++;
            }

            if (vDepthIdx[j].first > mpCurrentKeyFrame->mThDepth && nPoints > 300)
                break;
        }
    }
}

void LocalMapping::UpdateLocalMap()
{
    // Each map point vote for the keyframes
    // in which it has been observed
    map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < mpCurrentKeyFrame->mvpObservedMapPoints.size(); i++)
    {
        if (mpCurrentKeyFrame->mvpObservedMapPoints[i])
        {
            MapPoint *pMP = mpCurrentKeyFrame->mvpObservedMapPoints[i];
            if (!pMP->isBad())
            {
                const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (std::map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
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
    {
        mpReferenceKF = pKFmax;
    }

    // Update local map points
    mvpLocalMapPoints.clear();
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
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
}

void LocalMapping::MatchLocalPoints()
{
    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP->isBad())
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
        ORBmatcher matcher(0.8);
        nToMatch = matcher.SearchByProjection(mpCurrentKeyFrame, mvpLocalMapPoints, 3);
    }

    std::cout << "No. of KFs: " << mvpLocalKeyFrames.size()
              << " ;No. of MPs: " << mvpLocalMapPoints.size()
              << " ;No. of Matches: " << nToMatch << std::endl;

    mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    if (mpCurrentKeyFrame->mnId == 0)
        mvpRecentlyAddedMapPoints = mpCurrentKeyFrame->GetMapPointMatches();

    // Compute Bags of Words structures
    // mpCurrentKeyFrame->ComputeBoW();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::SetShouldQuit()
{
    mbShouldQuit = true;
}