#include "Mapping.h"
#include "Matcher.h"
#include "Bundler.h"

namespace SLAM
{

Mapping::Mapping(Map *pMap)
    : mpMap(pMap), mbShouldQuit(false), mpReferenceKF(NULL)
{
}

void Mapping::Run()
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

void Mapping::Kill()
{
    mbShouldQuit = true;
}

void Mapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for (vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
    {
        KeyFrame *pKF = *vit;
        if (pKF->mnId == 0)
            continue;
        const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

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
                        const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                        int nObs = 0;
                        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
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
}

void Mapping::CreateNewMapPoints()
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

void Mapping::UpdateLocalMap()
{
    // Each map point vote for the keyframes
    // in which it has been observed
    std::map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < mpCurrentKeyFrame->mvpObservedMapPoints.size(); i++)
    {
        MapPoint *pMP = mpCurrentKeyFrame->mvpObservedMapPoints[i];
        if (pMP && !pMP->isBad())
        {
            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
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
    // All points in the local map is included
    mvpLocalMapPoints.clear();
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        // Get map points in the keyframe
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

void Mapping::MatchLocalPoints()
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
        Matcher matcher(0.8);
        // Project points to the current keyframe
        // And search for potential corresponding points
        nToMatch = matcher.SearchByProjection(mpCurrentKeyFrame, mvpLocalMapPoints, 3);
    }

    // Update covisibility based on the correspondences
    mpCurrentKeyFrame->UpdateConnections();

    // std::cout << "No. of KFs: " << mvpLocalKeyFrames.size()
    //           << " ;No. of MPs: " << mvpLocalMapPoints.size()
    //           << " ;No. of Matches: " << nToMatch << std::endl;
}

void Mapping::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

void Mapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

bool Mapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

void Mapping::ProcessNewKeyFrame()
{
    {
        // Take the new keyframe out fromt the list
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

} // namespace SLAM