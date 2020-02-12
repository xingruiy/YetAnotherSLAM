#include "Mapping.h"
#include "Matcher.h"
#include "Bundler.h"

namespace SLAM
{

Mapping::Mapping(Map *map) : mpMap(map)
{
    ORBExtractor = new ORB_SLAM2::ORBextractor(g_ORBNFeatures, g_ORBScaleFactor, g_ORBNLevels, g_ORBIniThFAST, g_ORBMinThFAST);
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

            CreateNewMapPoints();

            UpdateConnections();
        }
    }

    std::cout << "Mapping Thread Killed." << std::endl;
}

void Mapping::reset()
{
    localKeyFrames = std::vector<KeyFrame *>();
    localMapPoints = std::vector<MapPoint *>();
}

void Mapping::addKeyFrameCandidate(Frame *F)
{
    std::unique_lock<std::mutex> lock(frameMutex);
    newFrameQueue.push_back(F);
}

bool Mapping::HasFrameToProcess()
{
    std::unique_lock<std::mutex> lock(frameMutex);
    return (!newFrameQueue.empty());
}

void Mapping::MakeNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(frameMutex);
        currentFrame = newFrameQueue.front();
        newFrameQueue.pop_front();
    }

    // Create new keyframe
    currentKeyFrame = new KeyFrame(currentFrame, mpMap, ORBExtractor);

    // Create map points for the first frame
    if (currentKeyFrame->mnId == 0)
    {
        for (int i = 0; i < currentKeyFrame->mvKeysUn.size(); ++i)
        {
            const float d = currentKeyFrame->mvDepth[i];
            if (d > 0)
            {
                auto posWorld = currentKeyFrame->UnprojectKeyPoint(i);
                MapPoint *pMP = new MapPoint(posWorld, mpMap, currentKeyFrame, i);
                pMP->AddObservation(currentKeyFrame, i);
                pMP->UpdateNormalAndDepth();
                pMP->ComputeDistinctiveDescriptors();

                currentKeyFrame->AddMapPoint(pMP, i);
                mpMap->AddMapPoint(pMP);
            }
        }
    }

    // Insert the keyframe in the map
    mpMap->AddKeyFrame(currentKeyFrame);
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
        if (currentKeyFrame->IsInFrustum(pMP, 0.5))
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
        nToMatch = matcher.SearchByProjection(currentKeyFrame, localMapPoints, 3);
    }

    // Update covisibility based on the correspondences
    currentKeyFrame->UpdateConnections();
}

void Mapping::KeyFrameCulling()
{
    auto vpLocalKeyFrames = currentKeyFrame->GetVectorCovisibleKeyFrames();
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
}

void Mapping::CreateNewMapPoints()
{
}

void Mapping::UpdateConnections()
{
    // Each map point vote for the keyframes
    // in which it has been observed
    std::map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < currentKeyFrame->mvpMapPoints.size(); i++)
    {
        MapPoint *pMP = currentKeyFrame->mvpMapPoints[i];
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
    {
        referenceKeyframe = pKFmax;
    }

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
            if (pMP->mnTrackReferenceForFrame == currentKeyFrame->mnId)
                continue;
            if (!pMP->isBad())
            {
                localMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = currentKeyFrame->mnId;
            }
        }
    }
}

} // namespace SLAM