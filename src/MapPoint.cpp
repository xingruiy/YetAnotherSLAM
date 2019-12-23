#include "MapPoint.h"
#include <cmath>

MapPoint::MapPoint(const Eigen::Vector3d &pos, KeyFrame *pRefKF, Map *pMap)
    : mpMap(pMap), mpRefKF(pRefKF), mWorldPos(pos)
{
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

MapPoint *MapPoint::GetReplaced()
{
    return mpReplaced;
}