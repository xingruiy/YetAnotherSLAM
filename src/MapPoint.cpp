#include "MapPoint.h"
#include <cmath>

std::mutex MapPoint::mGlobalMutex;
unsigned long MapPoint::nNextId = 0;

MapPoint::MapPoint(const Eigen::Vector3d &pos, Map *pMap, KeyFrame *pRefKF, const int &idxF)
    : mpMap(pMap), mpRefKF(pRefKF), mWorldPos(pos), nObs(0), mnVisible(1), mnFound(1)
{
    mnId = nNextId++;

    Eigen::Vector3d Ow = pRefKF->mTcw.matrix().topRightCorner(3, 1);
    mNormalVector = mWorldPos - Ow;
    mNormalVector.normalize();

    Eigen::Vector3d PC = pos - Ow;
    const float dist = PC.norm();
    const int level = pRefKF->mvKeys[idxF].octave;
    const float levelScaleFactor = pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    mfMaxDistance = dist * levelScaleFactor;
    mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];

    pRefKF->mDescriptors.row(idxF).copyTo(mDescriptor);
}

std::map<KeyFrame *, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}

void MapPoint::AddObservation(KeyFrame *pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
    nObs++;
}

void MapPoint::EraseObservation(KeyFrame *pKF)
{
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
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
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

void MapPoint::Replace(MapPoint *pMP)
{
    if (pMP->mnId == this->mnId)
        return;
}

Eigen::Vector3d MapPoint::GetNormal()
{
    unique_lock<mutex> lock2(mMutexPos);
    return mNormalVector;
}

MapPoint *MapPoint::GetReplaced()
{
    return mpReplaced;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
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
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pFrame->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pFrame->mnScaleLevels)
        nScale = pFrame->mnScaleLevels - 1;

    return nScale;
}