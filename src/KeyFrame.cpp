#include "KeyFrame.h"
#include "Converter.h"

unsigned long KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(const Frame &F, Map *pMap)
    : mpMap(pMap), mvKeys(F.mvKeys), mTcw(F.mTcw), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
      mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
      mvInvLevelSigma2(F.mvInvLevelSigma2), N(F.N), fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx),
      invfy(F.invfy), mvpMapPoints(F.mvpMapPoints), mvDepth(F.mvDepth), mvbOutlier(F.mvbOutlier),
      mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS), mfGridElementWidthInv(F.mfGridElementWidthInv),
      mfGridElementHeightInv(F.mfGridElementHeightInv), width(F.width), height(F.height), mbf(F.mbf),
      mThDepth(F.mThDepth), mvuRight(F.mvuRight), mDescriptors(F.mDescriptors), mpORBvocabulary(F.mpORBvocabulary)
{
  mnId = nNextId++;

  mGrid.resize(mnGridCols);
  for (int i = 0; i < mnGridCols; i++)
  {
    mGrid[i].resize(mnGridRows);
    for (int j = 0; j < mnGridRows; j++)
      mGrid[i][j] = F.mGrid[i][j];
  }
}

void KeyFrame::ComputeBoW()
{
  if (mBowVec.empty() || mFeatVec.empty())
  {
    vector<cv::Mat> vCurrentDesc = ORB_SLAM2::Converter::toDescriptorVector(mDescriptors);
    // Feature vector associate features with nodes in the 4th level (from leaves up)
    // We assume the vocabulary tree has 6 levels, change the 4 otherwise
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

vector<MapPoint *> KeyFrame::GetMapPointMatches()
{
  unique_lock<mutex> lock(mMutexFeatures);
  return mvpMapPoints;
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
{
  std::vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX = max(0, (int)floor((x - r) * mfGridElementWidthInv));
  if (nMinCellX >= FRAME_GRID_COLS)
    return vIndices;

  const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0)
    return vIndices;

  const int nMinCellY = max(0, (int)floor((y - r) * mfGridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS)
    return vIndices;

  const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0)
    return vIndices;

  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
  {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
    {
      const std::vector<size_t> vCell = mGrid[ix][iy];
      if (vCell.empty())
        continue;

      for (size_t j = 0, jend = vCell.size(); j < jend; j++)
      {
        const cv::KeyPoint &kpUn = mvKeys[vCell[j]];
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

  if (u < 0 || u > width)
    return false;
  if (v < 0 || v > height)
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