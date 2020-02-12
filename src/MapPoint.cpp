#include "MapPoint.h"
#include "Matcher.h"
#include <cmath>

namespace SLAM
{

std::mutex MapPoint::mGlobalMutex;
unsigned long MapPoint::nNextId = 0;

MapPoint::MapPoint(const Eigen::Vector3d &pos, KeyFrame *pRefKF, Map *pMap)
    : mpMap(pMap), mpRefKF(pRefKF), mWorldPos(pos), nObs(0), mnVisible(1), mnFound(1),
      mnTrackReferenceForFrame(-1), mpReplaced(static_cast<MapPoint *>(NULL)), mfMinDistance(0),
      mfMaxDistance(0), mnFuseCandidateForKF(0), mnFirstKFid(pRefKF->mnId)
{
    mNormalVector = Eigen::Vector3d::Zero();

    mnId = nNextId++;
}

MapPoint::MapPoint(const Eigen::Vector3d &pos, Map *pMap, KeyFrame *pRefKF, const int &idxF)
    : mpMap(pMap), mpRefKF(pRefKF), mWorldPos(pos), nObs(0), mnVisible(1), mnFound(1),
      mnTrackReferenceForFrame(-1), mpReplaced(static_cast<MapPoint *>(NULL)), mfMinDistance(0),
      mfMaxDistance(0), mnFuseCandidateForKF(0), mnFirstKFid(pRefKF->mnId)
{
    mnId = nNextId++;

    Eigen::Vector3d Ow = pRefKF->mTcw.matrix().topRightCorner(3, 1);
    mNormalVector = mWorldPos - Ow;
    mNormalVector.normalize();

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
            nObs--;
            mObservations.erase(pKF);

            if (mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if (nObs <= 2)
                bBad = true;
        }
    }

    if (bBad)
        SetBadFlag();
}

void MapPoint::SetBadFlag()
{
}

bool MapPoint::isBad()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    std::unique_lock<std::mutex> lock2(mMutexPos);
    return mbBad;
}

void MapPoint::Replace(MapPoint *pMP)
{
    if (pMP->mnId == this->mnId)
        return;
}

Eigen::Vector3d MapPoint::GetNormal()
{
    std::unique_lock<std::mutex> lock2(mMutexPos);
    return mNormalVector;
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

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
}

void MapPoint::UpdateNormalAndDepth()
{
    std::map<KeyFrame *, size_t> observations;
    KeyFrame *pRefKF;
    Eigen::Vector3d Pos;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        if (mbBad)
            return;
        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = mWorldPos;
    }

    if (observations.empty())
        return;

    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    int n = 0;
    for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;
        Eigen::Vector3d Owi = pKF->mTcw.translation();
        Eigen::Vector3d normali = mWorldPos - Owi;
        normal += normali.normalized();
        n++;
    }

    Eigen::Vector3d PC = Pos - pRefKF->mTcw.translation();
    const float dist = PC.norm();
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor = pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    {
        std::unique_lock<std::mutex> lock3(mMutexPos);
        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
        mNormalVector = normal / n;
    }
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
            int distij = Matcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
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
        sort(vDists.begin(), vDists.end());
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

int MapPoint::PredictScale(const float &currentDist, Frame *pFrame)
{
    // float ratio;
    // {
    //     std::unique_lock<std::mutex> lock(mMutexPos);
    //     ratio = mfMaxDistance / currentDist;
    // }

    // int nScale = ceil(log(ratio) / pFrame->mfLogScaleFactor);
    // if (nScale < 0)
    //     nScale = 0;
    // else if (nScale >= pFrame->mnScaleLevels)
    //     nScale = pFrame->mnScaleLevels - 1;

    // return nScale;
}

} // namespace SLAM