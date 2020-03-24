#include "MapPoint.h"
#include "ORBMatcher.h"
#include <cmath>

namespace SLAM
{

std::mutex MapPoint::mGlobalMutex;
unsigned long MapPoint::nNextId = 0;

MapPoint::MapPoint(const Eigen::Vector3d &pos, KeyFrame *pRefKF, Map *pMap)
    : mnFirstKFid(pRefKF->mnId), nObs(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
      mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
      mpReplaced(nullptr), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap), mWorldPos(pos),
      mAvgViewingDir(Eigen::Vector3d::Zero()), mnTrackReferenceForFrame(0), mnLastFrameSeen(pRefKF->mnFrameId)
{
    std::unique_lock<std::mutex> lock(pMap->mPointCreateMutex);
    mnId = nNextId++;
}

MapPoint::MapPoint(const Eigen::Vector3d &pos, Map *pMap, KeyFrame *pRefKF, const int &idxF)
    : mnFirstKFid(pRefKF->mnId), mpRefKF(pRefKF), nObs(0), mnBALocalForKF(0), mnFuseCandidateForKF(0),
      mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0),
      mnVisible(1), mnFound(1), mbBad(false), mpReplaced(nullptr), mpMap(pMap), mWorldPos(pos),
      mnTrackReferenceForFrame(0), mnLastFrameSeen(pRefKF->mnFrameId)
{
    {
        std::unique_lock<std::mutex> lock(pMap->mPointCreateMutex);
        mnId = nNextId++;
    }

    Eigen::Vector3d Ow = pRefKF->GetPose().matrix().topRightCorner(3, 1);
    mAvgViewingDir = mWorldPos - Ow;
    mAvgViewingDir.normalize();

    Eigen::Vector3d PC = pos - Ow;
    const float dist = PC.norm();
    const int level = pRefKF->mvKeysUn[idxF].octave;
    const float levelScaleFactor = pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    mfMaxDistance = dist * levelScaleFactor;
    mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];

    pRefKF->mDescriptors.row(idxF).copyTo(mDescriptor);
}

std::map<KeyFrame *, size_t> MapPoint::GetObservations()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return nObs;
}

cv::Mat MapPoint::GetDescriptor()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

void MapPoint::IncreaseFound(int n)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mnFound += n;
}

void MapPoint::IncreaseVisible(int n)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mnVisible += n;
}

void MapPoint::AddObservation(KeyFrame *pKF, size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;

    if (pKF->mvuRight[idx] >= 0)
        nObs += 2;
    else
        nObs++;
}

void MapPoint::EraseObservation(KeyFrame *pKF)
{
    bool bBad = false;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if (pKF->mvuRight[idx] >= 0)
                nObs -= 2;
            else
                nObs--;

            mObservations.erase(pKF);

            if (mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            if (nObs <= 2) // If only 2 observations or less, discard point
                bBad = true;
        }
    }

    if (bBad)
        SetBadFlag();
}

void MapPoint::SetBadFlag()
{
    mbBad = true;
}

bool MapPoint::isBad()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    std::unique_lock<std::mutex> lock2(mMutexPos);
    return mbBad;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

void MapPoint::Replace(MapPoint *pMP)
{
    if (pMP->mnId == this->mnId)
        return;

    int nvisible, nfound;
    std::map<KeyFrame *, size_t> obs;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        obs = mObservations;
        mObservations.clear();
        mbBad = true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for (auto mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame *pKF = mit->first;

        if (!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            pMP->AddObservation(pKF, mit->second);
        }
        else
        {
            pKF->EraseMapPointMatch(mit->second);
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

Eigen::Vector3d MapPoint::GetNormal()
{
    std::unique_lock<std::mutex> lock2(mMutexPos);
    return mAvgViewingDir;
}

KeyFrame *MapPoint::GetReferenceKeyFrame()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapPoint::SetWorldPos(const Eigen::Vector3d &pos)
{
    std::unique_lock<std::mutex> lock2(mGlobalMutex);
    std::unique_lock<std::mutex> lock(mMutexPos);
    mWorldPos = pos;
}

Eigen::Vector3d MapPoint::GetWorldPos()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mWorldPos;
}

MapPoint *MapPoint::GetReplaced()
{
    return mpReplaced;
}

float MapPoint::GetFoundRatio()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound) / mnVisible;
}

void MapPoint::UpdateNormalAndDepth()
{
    std::map<KeyFrame *, size_t> Obs;
    KeyFrame *pRefKF;
    Eigen::Vector3d Pos;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        if (mbBad)
            return;
        Obs = mObservations;
        pRefKF = mpRefKF;
        Pos = mWorldPos;
    }

    if (Obs.empty())
        return;

    Eigen::Vector3d viewingDir = Eigen::Vector3d::Zero();
    int n = 0;
    for (auto mit = Obs.begin(), mend = Obs.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;
        Eigen::Vector3d Owi = pKF->GetTranslation();
        Eigen::Vector3d viewingDiri = mWorldPos - Owi;
        viewingDir += viewingDiri.normalized();
        n++;
    }

    Eigen::Vector3d PC = Pos - pRefKF->GetTranslation();
    const float dist = PC.norm();
    const int level = pRefKF->mvKeysUn[Obs[pRefKF]].octave;
    const float levelScaleFactor = pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    {
        std::unique_lock<std::mutex> lock3(mMutexPos);
        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
        mAvgViewingDir = viewingDir / n;
    }
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    std::vector<cv::Mat> vDescriptors;
    std::map<KeyFrame *, size_t> observations;

    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        if (mbBad)
            return;
        observations = mObservations;
    }

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;

        if (!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if (vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for (size_t i = 0; i < N; i++)
    {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++)
        {
            int distij = ORBMatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for (size_t i = 0; i < N; i++)
    {
        std::vector<int> vDists(Distances[i], Distances[i] + N);
        std::sort(vDists.begin(), vDists.end());
        int median = vDists[0.5 * (N - 1)];

        if (median < BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF)
{
    float ratio;
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels - 1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame *pF)
{
    float ratio;
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pF->mnScaleLevels)
        nScale = pF->mnScaleLevels - 1;

    return nScale;
}

} // namespace SLAM