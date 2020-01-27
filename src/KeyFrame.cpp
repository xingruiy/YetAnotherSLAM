#include "KeyFrame.h"
#include "Converter.h"

unsigned long KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(const Frame &F, Map *pMap)
    : mpMap(pMap), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mTcw(F.mTcw), mnScaleLevels(F.mnScaleLevels), mImGray(F.mImGray),
      mfScaleFactor(F.mfScaleFactor), mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors),
      mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2), N(F.N), fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy),
      invfx(F.invfx), invfy(F.invfy), mvpMapPoints(F.mvpMapPoints), mvDepth(F.mvDepth), mvbOutlier(F.mvbOutlier),
      mbBad(false), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS), mfGridElementWidthInv(F.mfGridElementWidthInv),
      mfGridElementHeightInv(F.mfGridElementHeightInv), mbf(F.mbf), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
      mnMaxY(F.mnMaxY), mThDepth(F.mThDepth), mvuRight(F.mvuRight), mDescriptors(F.mDescriptors), mpORBvocabulary(F.mpORBvocabulary),
      mbFirstConnection(true)
{
  mnId = nNextId++;

  // Copy feature point grid
  mGrid.resize(mnGridCols);
  for (int i = 0; i < mnGridCols; i++)
  {
    mGrid[i].resize(mnGridRows);
    for (int j = 0; j < mnGridRows; j++)
      mGrid[i][j] = F.mGrid[i][j];
  }
}

bool KeyFrame::isBad()
{
  unique_lock<mutex> lock(mMutexConnections);
  return mbBad;
}

vector<MapPoint *> KeyFrame::GetMapPointMatches()
{
  unique_lock<mutex> lock(mMutexFeatures);
  return mvpMapPoints;
}

Eigen::Vector3d KeyFrame::UnprojectKeyPoint(int i)
{
  const float z = mvDepth[i];

  if (z > 0)
  {
    const float u = mvKeysUn[i].pt.x;
    const float v = mvKeysUn[i].pt.y;
    const float x = (u - cx) * z * invfx;
    const float y = (v - cy) * z * invfy;
    Eigen::Vector3d x3Dc(x, y, z);

    unique_lock<mutex> lock(mMutexPose);
    return mTcw * x3Dc;
  }
  else
  {
    return Eigen::Vector3d(0, 0, 0);
  }
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
{
  std::vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
  if (nMinCellX >= FRAME_GRID_COLS)
    return vIndices;

  const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0)
    return vIndices;

  const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS)
    return vIndices;

  const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0)
    return vIndices;

  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
  {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
    {
      const vector<size_t> vCell = mGrid[ix][iy];
      if (vCell.empty())
        continue;

      for (size_t j = 0, jend = vCell.size(); j < jend; j++)
      {
        const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];

        if (bCheckLevels)
        {
          if (kpUn.octave < minLevel)
            continue;
          if (maxLevel >= 0)
            if (kpUn.octave > maxLevel)
              continue;
        }

        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r)
          vIndices.push_back(vCell[j]);
      }
    }
  }

  return vIndices;
}

bool KeyFrame::IsInFrustum(MapPoint *pMP, float viewingCosLimit)
{
  pMP->mbTrackInView = false;

  // 3D in camera coordinates
  Eigen::Vector3d Pc = mTcw.inverse() * pMP->mWorldPos;
  const float &PcX = Pc(0);
  const float &PcY = Pc(1);
  const float &PcZ = Pc(2);

  // Check positive depth
  if (PcZ < 0.0f)
    return false;

  // Project in image and check it is not outside
  const float invz = 1.0f / PcZ;
  const float u = fx * PcX * invz + cx;
  const float v = fy * PcY * invz + cy;

  if (u < mnMinX || u > mnMaxX || v < mnMinY || v > mnMaxY)
    return false;
  // Check distance is in the scale invariance region of the MapPoint
  const float maxDistance = pMP->GetMaxDistanceInvariance();
  const float minDistance = pMP->GetMinDistanceInvariance();
  const Eigen::Vector3d PO = pMP->mWorldPos - mTcw.translation();
  const float dist = PO.norm();

  if (dist < minDistance || dist > maxDistance)
    return false;

  // Check viewing angle
  Eigen::Vector3d Pn = pMP->GetNormal();

  const float viewCos = PO.dot(Pn) / dist;

  if (viewCos < viewingCosLimit)
    return false;

  // Predict scale in the image
  const int nPredictedLevel = pMP->PredictScale(dist, this);

  // Data used by the tracking
  pMP->mbTrackInView = true;
  pMP->mTrackProjX = u;
  pMP->mTrackProjXR = u - mbf * invz;
  pMP->mTrackProjY = v;
  pMP->mnTrackScaleLevel = nPredictedLevel;
  pMP->mTrackViewCos = viewCos;

  return true;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  mvpMapPoints[idx] = pMP;
}

void KeyFrame::UpdateConnections()
{
  std::map<KeyFrame *, int> KFcounter;
  vector<MapPoint *> vpMP;

  {
    unique_lock<mutex> lockMPs(mMutexFeatures);
    vpMP = mvpMapPoints;
  }

  //For all map points in keyframe check in which other keyframes are they seen
  //Increase counter for those keyframes
  for (vector<MapPoint *>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
  {
    MapPoint *pMP = *vit;

    if (!pMP || pMP->isBad())
      continue;

    map<KeyFrame *, size_t> observations = pMP->GetObservations();
    for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
      if (mit->first->mnId == mnId)
        continue;
      KFcounter[mit->first]++;
    }
  }

  // This should not happen
  if (KFcounter.empty())
    return;

  //If the counter is greater than threshold add connection
  //In case no keyframe counter is over threshold add the one with maximum counter
  int nmax = 0;
  KeyFrame *pKFmax = NULL;
  int th = 15;

  vector<pair<int, KeyFrame *>> vPairs;
  vPairs.reserve(KFcounter.size());
  for (map<KeyFrame *, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
  {
    if (mit->second > nmax)
    {
      nmax = mit->second;
      pKFmax = mit->first;
    }
    if (mit->second >= th)
    {
      vPairs.push_back(make_pair(mit->second, mit->first));
      (mit->first)->AddConnection(this, mit->second);
    }
  }

  if (vPairs.empty())
  {
    vPairs.push_back(make_pair(nmax, pKFmax));
    pKFmax->AddConnection(this, nmax);
  }

  sort(vPairs.begin(), vPairs.end());
  list<KeyFrame *> lKFs;
  list<int> lWs;
  for (size_t i = 0; i < vPairs.size(); i++)
  {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  {
    unique_lock<mutex> lockCon(mMutexConnections);

    mConnectedKeyFrameWeights = KFcounter;
    mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

    if (mbFirstConnection && mnId != 0)
    {
      mpParent = mvpOrderedConnectedKeyFrames.front();
      mpParent->AddChild(this);
      mbFirstConnection = false;
    }
  }
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
  {
    unique_lock<mutex> lock(mMutexConnections);
    if (!mConnectedKeyFrameWeights.count(pKF))
      mConnectedKeyFrameWeights[pKF] = weight;
    else if (mConnectedKeyFrameWeights[pKF] != weight)
      mConnectedKeyFrameWeights[pKF] = weight;
    else
      return;
  }

  UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
  unique_lock<mutex> lock(mMutexConnections);
  vector<pair<int, KeyFrame *>> vPairs;
  vPairs.reserve(mConnectedKeyFrameWeights.size());
  for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
    vPairs.push_back(make_pair(mit->second, mit->first));

  sort(vPairs.begin(), vPairs.end());
  list<KeyFrame *> lKFs;
  list<int> lWs;
  for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
  {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
  mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
  unique_lock<mutex> lockCon(mMutexConnections);
  mspChildrens.insert(pKF);
}
