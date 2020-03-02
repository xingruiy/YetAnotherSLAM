#include "KeyFrame.h"
#include "Converter.h"
#include "GlobalDef.h"

namespace SLAM
{

unsigned long KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(const Frame &F, Map *pMap)
    : mpMap(pMap), mTcw(F.mTcw), mpExtractor(F.mpORBextractor), mRelativePose(F.mRelativePose),
      mbBad(false), mbToBeErased(false), mbNotErase(false), mbFirstConnection(true),
      mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mImg(F.mImGray.clone()),
      mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0),
      fx(g_fx[0]), fy(g_fy[0]), cx(g_cx[0]), cy(g_cy[0]), mK(g_cvCalib),
      invfx(g_invfx[0]), invfy(g_invfy[0]), mbf(g_bf), mThDepth(g_thDepth),
      mbVoxelStructMarginalized(false), mTimeStamp(F.mTimeStamp), mpParent(nullptr)
{
  mnId = nNextId++;

  // camera baseline
  mb = mbf / fx;

  mGrid.resize(FRAME_GRID_COLS);
  for (int i = 0; i < FRAME_GRID_COLS; i++)
    mGrid[i].resize(FRAME_GRID_ROWS);

  (*mpExtractor)(F.mImGray, cv::Mat(), mvKeys, mDescriptors);
  N = mvKeys.size();
  mvbOutlier.resize(N, false);
  mvpMapPoints.resize(N, static_cast<MapPoint *>(NULL));

  UndistortKeys();
  AssignFeaturesToGrid();
  ComputeStereoRGBD(F.mImDepth);

  // Scale Level Info
  mnScaleLevels = mpExtractor->GetLevels();
  mfScaleFactor = mpExtractor->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor); // For predicting scales
  mvScaleFactors = mpExtractor->GetScaleFactors();
  mvLevelSigma2 = mpExtractor->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpExtractor->GetInverseScaleSigmaSquares();
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

void KeyFrame::ComputeBoW(ORBVocabulary *voc)
{
  if (mBowVec.empty())
  {
    std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    voc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

void KeyFrame::SetPose(const Sophus::SE3d &Tcw)
{
  mTcw = Tcw;

  if (mpVoxelStruct)
    mpVoxelStruct->mTcw = Tcw;
}

Sophus::SE3d KeyFrame::GetPose()
{
  return mTcw;
}

Sophus::SE3d KeyFrame::GetPoseInverse()
{
  return mTcw.inverse();
}

Eigen::Matrix3d KeyFrame::GetRotation()
{
  return mTcw.rotationMatrix();
}

Eigen::Vector3d KeyFrame::GetTranslation()
{
  return mTcw.translation();
}

void KeyFrame::EraseMapPointMatch(MapPoint *pMP)
{
  int idx = pMP->GetIndexInKeyFrame(this);
  if (idx >= 0)
    mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
  mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
}

void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP)
{
  mvpMapPoints[idx] = pMP;
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

  // Keypoint's coordinates are undistorted, which could cause to go out of the image
  if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
    return false;

  return true;
}

void KeyFrame::ComputeStereoRGBD(const cv::Mat depth_image)
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

    if (v < 1 && u < 1 && v >= g_width[0] - 1 && u >= g_height[0] - 1)
      continue;

    const float d = depth_image.at<float>(v, u);
    if (d > 0)
    {
      mvDepth[i] = d;
      mvuRight[i] = kpU.pt.x - g_bf / d;
    }
  }
}

bool KeyFrame::isBad()
{
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mbBad;
}

std::vector<MapPoint *> KeyFrame::GetMapPointMatches()
{
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return mvpMapPoints;
}

MapPoint *KeyFrame::GetMapPoint(const size_t &idx)
{
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return mvpMapPoints[idx];
}

std::set<MapPoint *> KeyFrame::GetMapPoints()
{
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  std::set<MapPoint *> s;
  for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++)
  {
    if (!mvpMapPoints[i])
      continue;
    MapPoint *pMP = mvpMapPoints[i];
    if (!pMP->isBad())
      s.insert(pMP);
  }
  return s;
}

bool KeyFrame::UnprojectKeyPoint(Eigen::Vector3d &posWorld, const int &i)
{
  const float z = mvDepth[i];

  if (z > 0)
  {
    const float u = mvKeysUn[i].pt.x;
    const float v = mvKeysUn[i].pt.y;
    const float x = (u - cx) * z * invfx;
    const float y = (v - cy) * z * invfy;
    std::unique_lock<std::mutex> lock(poseMutex);
    posWorld = mTcw * Eigen::Vector3d(x, y, z);

    return true;
  }

  return false;
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

    std::unique_lock<std::mutex> lock(poseMutex);
    return mTcw * x3Dc;
  }
  else
  {
    return Eigen::Vector3d(0, 0, 0);
  }
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, float r, int minLevel, int maxLevel)
{
  std::vector<size_t> vIndices;
  vIndices.reserve(N);

  int nMinCellX = std::max(0, (int)floor((x - g_minX - r) * g_gridElementWidthInv));
  if (nMinCellX >= FRAME_GRID_COLS)
    return vIndices;

  int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - g_minX + r) * g_gridElementWidthInv));
  if (nMaxCellX < 0)
    return vIndices;

  int nMinCellY = std::max(0, (int)floor((y - g_minY - r) * g_gridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS)
    return vIndices;

  int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - g_minY + r) * g_gridElementHeightInv));
  if (nMaxCellY < 0)
    return vIndices;

  bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
  {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
    {
      std::vector<size_t> vCell = mGrid[ix][iy];
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

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
  return (x >= g_minX && x < g_maxX && y >= g_minY && y < g_maxY);
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

  if (u < g_minX || u > g_maxX || v < g_minY || v > g_maxY)
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
  std::vector<MapPoint *> vpMP;

  {
    std::unique_lock<std::mutex> lockMPs(mMutexFeatures);
    vpMP = mvpMapPoints;
  }

  //For all map points in keyframe check in which other keyframes are they seen
  //Increase counter for those keyframes
  for (auto vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
  {
    MapPoint *pMP = *vit;

    if (!pMP || pMP->isBad())
      continue;

    std::map<KeyFrame *, size_t> observations = pMP->GetObservations();
    for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
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

  std::vector<std::pair<int, KeyFrame *>> vPairs;
  vPairs.reserve(KFcounter.size());
  for (auto mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
  {
    if (mit->second > nmax)
    {
      nmax = mit->second;
      pKFmax = mit->first;
    }
    if (mit->second >= th)
    {
      vPairs.push_back(std::make_pair(mit->second, mit->first));
      (mit->first)->AddConnection(this, mit->second);
    }
  }

  if (vPairs.empty())
  {
    vPairs.push_back(std::make_pair(nmax, pKFmax));
    pKFmax->AddConnection(this, nmax);
  }

  sort(vPairs.begin(), vPairs.end());
  std::list<KeyFrame *> lKFs;
  std::list<int> lWs;
  for (size_t i = 0; i < vPairs.size(); i++)
  {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  {
    std::unique_lock<std::mutex> lockCon(mMutexConnections);

    mConnectedKeyFrameWeights = KFcounter;
    mvpOrderedConnectedKeyFrames = std::vector<KeyFrame *>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

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
    std::unique_lock<std::mutex> lock(mMutexConnections);
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
  std::unique_lock<std::mutex> lock(mMutexConnections);
  std::vector<std::pair<int, KeyFrame *>> vPairs;
  vPairs.reserve(mConnectedKeyFrameWeights.size());
  for (auto mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
    vPairs.push_back(std::make_pair(mit->second, mit->first));

  sort(vPairs.begin(), vPairs.end());
  std::list<KeyFrame *> lKFs;
  std::list<int> lWs;
  for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
  {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  mvpOrderedConnectedKeyFrames = std::vector<KeyFrame *>(lKFs.begin(), lKFs.end());
  mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  mspChildrens.insert(pKF);
}

std::set<KeyFrame *> KeyFrame::GetChilds()
{
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  return mspChildrens;
}

std::set<KeyFrame *> KeyFrame::GetConnectedKeyFrames()
{
  std::unique_lock<std::mutex> lock(mMutexConnections);
  std::set<KeyFrame *> s;
  for (auto mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
    s.insert(mit->first);
  return s;
}

std::vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames()
{
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mvpOrderedConnectedKeyFrames;
}

std::vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
  std::unique_lock<std::mutex> lock(mMutexConnections);
  if ((int)mvpOrderedConnectedKeyFrames.size() < N)
    return mvpOrderedConnectedKeyFrames;
  else
    return std::vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
}

std::vector<KeyFrame *> KeyFrame::GetCovisiblesByWeight(const int &w)
{
  std::unique_lock<std::mutex> lock(mMutexConnections);

  if (mvpOrderedConnectedKeyFrames.empty())
    return std::vector<KeyFrame *>();

  auto it = std::upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, [&](int a, int b) { return a >= b; });
  if (it == mvOrderedWeights.end())
    return std::vector<KeyFrame *>();
  else
  {
    int n = it - mvOrderedWeights.begin();
    return std::vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
  }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
  std::unique_lock<std::mutex> lock(mMutexConnections);
  if (mConnectedKeyFrameWeights.count(pKF))
    return mConnectedKeyFrameWeights[pKF];
  else
    return 0;
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  mpParent = pKF;
  pKF->AddChild(this);
}

void KeyFrame::SetBadFlag()
{
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (mnId == 0)
      return;
    else if (mbNotErase)
    {
      mbToBeErased = true;
      return;
    }
  }

  for (auto mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
    mit->first->EraseConnection(this);

  for (size_t i = 0; i < mvpMapPoints.size(); i++)
    if (mvpMapPoints[i])
      mvpMapPoints[i]->EraseObservation(this);
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    std::unique_lock<std::mutex> lock1(mMutexFeatures);

    mConnectedKeyFrameWeights.clear();
    mvpOrderedConnectedKeyFrames.clear();

    // Update Spanning Tree
    std::set<KeyFrame *> sParentCandidates;
    sParentCandidates.insert(mpParent);

    // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
    // Include that children as new parent candidate for the rest
    while (!mspChildrens.empty())
    {
      bool bContinue = false;

      int max = -1;
      KeyFrame *pC;
      KeyFrame *pP;

      for (auto sit = mspChildrens.begin(), send = mspChildrens.end(); sit != send; sit++)
      {
        KeyFrame *pKF = *sit;
        if (pKF->isBad())
          continue;

        // Check if a parent candidate is connected to the keyframe
        std::vector<KeyFrame *> vpConnected = pKF->GetVectorCovisibleKeyFrames();
        for (size_t i = 0, iend = vpConnected.size(); i < iend; i++)
        {
          for (auto spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
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
      for (auto sit = mspChildrens.begin(); sit != mspChildrens.end(); sit++)
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

KeyFrame *KeyFrame::GetParent()
{
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  mbNotErase = true;
  mspLoopEdges.insert(pKF);
}

std::set<KeyFrame *> KeyFrame::GetLoopEdges()
{
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  return mspLoopEdges;
}

void KeyFrame::EraseConnection(KeyFrame *pKF)
{
  bool bUpdate = false;
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF))
    {
      mConnectedKeyFrameWeights.erase(pKF);
      bUpdate = true;
    }
  }

  if (bUpdate)
    UpdateBestCovisibles();
}

void KeyFrame::SetNotErase()
{
  std::unique_lock<std::mutex> lock(mMutexConnections);
  mbNotErase = true;
}

void KeyFrame::SetErase()
{
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (mspLoopEdges.empty())
    {
      mbNotErase = false;
    }
  }

  if (mbToBeErased)
  {
    SetBadFlag();
  }
}

} // namespace SLAM