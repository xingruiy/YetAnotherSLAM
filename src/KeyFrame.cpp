#include "KeyFrame.h"
// #include "Converter.h"

namespace SLAM
{

unsigned long KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(Frame *F, Map *map, ORB_SLAM2::ORBextractor *pExtractor)
    : mpMap(map), mTcw(F->mTcw)
{
  mnId = nNextId++;

  (*pExtractor)(F->mImGray, cv::Mat(), mvKeys, mDescriptors);
  N = mvKeys.size();
  UndistortKeys();

  ComputeDepth(F->mImDepth);

  // Scale Level Info
  mnScaleLevels = pExtractor->GetLevels();
  mfScaleFactor = pExtractor->GetScaleFactor();
  // For predicting scales
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = pExtractor->GetScaleFactors();
  mvLevelSigma2 = pExtractor->GetScaleSigmaSquares();
  mvInvLevelSigma2 = pExtractor->GetInverseScaleSigmaSquares();
}

void KeyFrame::UndistortKeys()
{
  if (N == 0)
    return;

  if (g_distCoeff.at<float>(0) == 0.0)
  {
    mvKeysUn = mvKeys;
    return;
  }

  // Fill matrix with points
  cv::Mat mat(N, 2, CV_32F);
  for (int i = 0; i < N; i++)
  {
    mat.at<float>(i, 0) = mvKeys[i].pt.x;
    mat.at<float>(i, 1) = mvKeys[i].pt.y;
  }

  // Undistort points
  mat = mat.reshape(2);
  cv::undistortPoints(mat, mat, g_cvCalib, g_distCoeff, cv::Mat(), g_cvCalib);
  mat = mat.reshape(1);

  // Fill undistorted keypoint vector
  mvKeysUn.resize(N);
  for (int i = 0; i < N; i++)
  {
    cv::KeyPoint kp = mvKeys[i];
    kp.pt.x = mat.at<float>(i, 0);
    kp.pt.y = mat.at<float>(i, 1);
    mvKeysUn[i] = kp;
  }
}

void KeyFrame::AssignFeaturesToGrid()
{
  for (int i = 0; i < N; i++)
  {
    const cv::KeyPoint &kp = mvKeysUn[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY))
      mGrid[nGridPosX][nGridPosY].push_back(i);
  }
}

bool KeyFrame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
  posX = round((kp.pt.x - g_minX) * g_gridElementWidthInv);
  posY = round((kp.pt.y - g_minY) * g_gridElementHeightInv);

  //Keypoint's coordinates are undistorted, which could cause to go out of the image
  if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
    return false;

  return true;
}

void KeyFrame::ComputeDepth(const cv::Mat depth_image)
{
  // The virtual right coordinate
  mvuRight = std::vector<float>(N, -1);
  // Key point depth
  mvDepth = std::vector<float>(N, -1);

  for (int i = 0; i < N; i++)
  {
    const cv::KeyPoint &kp = mvKeys[i];
    const cv::KeyPoint &kpU = mvKeysUn[i];

    const float &v = kp.pt.y;
    const float &u = kp.pt.x;

    const float d = depth_image.at<float>(v, u);

    if (d > 0)
    {
      mvDepth[i] = d;
      mvuRight[i] = kpU.pt.x - g_bf / d;
    }
  }
}

KeyFrame::KeyFrame(Frame &F, Map *pMap)
// : mpMap(pMap), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mTcw(F.mTcw), mnScaleLevels(F.mnScaleLevels), mImGray(F.mImGray),
//   mfScaleFactor(F.mfScaleFactor), mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors),
//   mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2), N(F.N), fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy),
//   invfx(F.invfx), invfy(F.invfy), mvpMapPoints(F.mvpMapPoints), mvDepth(F.mvDepth), mvbOutlier(F.mvbOutlier),
//   mbBad(false), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS), mbf(F.mbf), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
//   mnMaxY(F.mnMaxY), mThDepth(F.mThDepth), mvuRight(F.mvuRight), mDescriptors(F.mDescriptors), mpORBvocabulary(F.mpORBvocabulary),
//   mbFirstConnection(true)
{
  mnId = nNextId++;

  // Copy feature point grid
  // mGrid.resize(mnGridCols);
  // for (int i = 0; i < mnGridCols; i++)
  // {
  //   mGrid[i].resize(mnGridRows);
  //   for (int j = 0; j < mnGridRows; j++)
  //     mGrid[i][j] = F.mGrid[i][j];
  // }
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
  float z = mvDepth[i];

  if (z > 0)
  {
    float u = mvKeysUn[i].pt.x;
    float v = mvKeysUn[i].pt.y;
    float x = (u - cx) * z * invfx;
    float y = (v - cy) * z * invfy;
    Eigen::Vector3d x3Dc(x, y, z);

    unique_lock<mutex> lock(mMutexPose);
    return mTcw * x3Dc;
  }
  else
  {
    return Eigen::Vector3d(0, 0, 0);
  }
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(float &x, float &y, float &r, int minLevel, int maxLevel)
{
  std::vector<size_t> vIndices;
  vIndices.reserve(N);

  int nMinCellX = max(0, (int)floor((x - mnMinX - r) * g_gridElementWidthInv));
  if (nMinCellX >= FRAME_GRID_COLS)
    return vIndices;

  int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * g_gridElementWidthInv));
  if (nMaxCellX < 0)
    return vIndices;

  int nMinCellY = max(0, (int)floor((y - mnMinY - r) * g_gridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS)
    return vIndices;

  int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * g_gridElementHeightInv));
  if (nMaxCellY < 0)
    return vIndices;

  bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
  {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
    {
      vector<size_t> vCell = mGrid[ix][iy];
      if (vCell.empty())
        continue;

      for (size_t j = 0, jend = vCell.size(); j < jend; j++)
      {
        cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];

        if (bCheckLevels)
        {
          if (kpUn.octave < minLevel)
            continue;
          if (maxLevel >= 0)
            if (kpUn.octave > maxLevel)
              continue;
        }

        float distx = kpUn.pt.x - x;
        float disty = kpUn.pt.y - y;

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
  float invz = 1.0f / PcZ;
  float u = fx * PcX * invz + cx;
  float v = fy * PcY * invz + cy;

  if (u < mnMinX || u > mnMaxX || v < mnMinY || v > mnMaxY)
    return false;
  // Check distance is in the scale invariance region of the MapPoint
  float maxDistance = pMP->GetMaxDistanceInvariance();
  float minDistance = pMP->GetMinDistanceInvariance();
  Eigen::Vector3d PO = pMP->mWorldPos - mTcw.translation();
  float dist = PO.norm();

  if (dist < minDistance || dist > maxDistance)
    return false;

  // Check viewing angle
  Eigen::Vector3d Pn = pMP->GetNormal();

  float viewCos = PO.dot(Pn) / dist;

  if (viewCos < viewingCosLimit)
    return false;

  // Predict scale in the image
  int nPredictedLevel = pMP->PredictScale(dist, this);

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

void KeyFrame::AddConnection(KeyFrame *pKF, int &weight)
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

std::vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames()
{
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mvpOrderedConnectedKeyFrames;
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
  unique_lock<mutex> lock(mMutexConnections);
  if (mConnectedKeyFrameWeights.count(pKF))
    return mConnectedKeyFrameWeights[pKF];
  else
    return 0;
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
  unique_lock<mutex> lockCon(mMutexConnections);
  mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
  unique_lock<mutex> lockCon(mMutexConnections);
  mpParent = pKF;
  pKF->AddChild(this);
}

void KeyFrame::SetBadFlag()
{
  {
    unique_lock<mutex> lock(mMutexConnections);
    if (mnId == 0)
      return;
    else if (mbNotErase)
    {
      mbToBeErased = true;
      return;
    }
  }

  for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
    mit->first->EraseConnection(this);

  for (size_t i = 0; i < mvpMapPoints.size(); i++)
    if (mvpMapPoints[i])
      mvpMapPoints[i]->EraseObservation(this);
  {
    unique_lock<mutex> lock(mMutexConnections);
    unique_lock<mutex> lock1(mMutexFeatures);

    mConnectedKeyFrameWeights.clear();
    mvpOrderedConnectedKeyFrames.clear();

    // Update Spanning Tree
    set<KeyFrame *> sParentCandidates;
    sParentCandidates.insert(mpParent);

    // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
    // Include that children as new parent candidate for the rest
    while (!mspChildrens.empty())
    {
      bool bContinue = false;

      int max = -1;
      KeyFrame *pC;
      KeyFrame *pP;

      for (set<KeyFrame *>::iterator sit = mspChildrens.begin(), send = mspChildrens.end(); sit != send; sit++)
      {
        KeyFrame *pKF = *sit;
        if (pKF->isBad())
          continue;

        // Check if a parent candidate is connected to the keyframe
        std::vector<KeyFrame *> vpConnected = pKF->GetVectorCovisibleKeyFrames();
        for (size_t i = 0, iend = vpConnected.size(); i < iend; i++)
        {
          for (std::set<KeyFrame *>::iterator spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
          {
            if (vpConnected[i]->mnId == (*spcit)->mnId)
            {
              int w = pKF->GetWeight(vpConnected[i]);
              if (w > max)
              {
                pC = pKF;
                pP = vpConnected[i];
                max = w;
                bContinue = true;
              }
            }
          }
        }
      }

      if (bContinue)
      {
        pC->ChangeParent(pP);
        sParentCandidates.insert(pC);
        mspChildrens.erase(pC);
      }
      else
        break;
    }

    // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
    if (!mspChildrens.empty())
      for (set<KeyFrame *>::iterator sit = mspChildrens.begin(); sit != mspChildrens.end(); sit++)
      {
        (*sit)->ChangeParent(mpParent);
      }

    mpParent->EraseChild(this);
    // mTcp = Tcw * mpParent->mTcw.inverse();
    mbBad = true;
  }

  mpMap->EraseKeyFrame(this);
  // mpKeyFrameDB->erase(this);
}

void KeyFrame::EraseConnection(KeyFrame *pKF)
{
  bool bUpdate = false;
  {
    unique_lock<mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF))
    {
      mConnectedKeyFrameWeights.erase(pKF);
      bUpdate = true;
    }
  }

  if (bUpdate)
    UpdateBestCovisibles();
}

} // namespace SLAM