#include "Frame.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

namespace slam
{

bool Frame::mbInitialComputations = true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{
}

Frame::Frame(const Frame &frame)
    : OrbVoc(frame.OrbVoc), OrbExt(frame.OrbExt),
      mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
      mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
      mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
      mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
      mDescriptors(frame.mDescriptors.clone()), mTcw(frame.mTcw), mTcp(frame.mTcp),
      pointsMatches(frame.pointsMatches), outlierFlag(frame.outlierFlag), meta(frame.meta),
      mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
      mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
      mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
      mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
  for (int i = 0; i < FRAME_GRID_COLS; i++)
    for (int j = 0; j < FRAME_GRID_ROWS; j++)
      mGrid[i][j] = frame.mGrid[i][j];
}

Frame::Frame(cv::Mat img, cv::Mat depth, ORBextractor *ext, ORBVocabulary *voc)
    : OrbVoc(voc), OrbExt(ext), mK(g_cvCalib.clone()),
      mDistCoef(g_distCoeff.clone()), mbf(g_bf), mThDepth(g_thDepth),
      mImGray(img.clone()), mImDepth(depth.clone())
{
  // Scale Level Info
  mnScaleLevels = OrbExt->GetLevels();
  mfScaleFactor = OrbExt->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = OrbExt->GetScaleFactors();
  mvInvScaleFactors = OrbExt->GetInverseScaleFactors();
  mvLevelSigma2 = OrbExt->GetScaleSigmaSquares();
  mvInvLevelSigma2 = OrbExt->GetInverseScaleSigmaSquares();

  // This is done only for the first Frame (or after a change in the calibration)
  if (mbInitialComputations)
  {
    ComputeImageBounds(img);

    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

    fx = mK.at<float>(0, 0);
    fy = mK.at<float>(1, 1);
    cx = mK.at<float>(0, 2);
    cy = mK.at<float>(1, 2);
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  // Has to be done at the very end
  mb = mbf / fx;
}

void Frame::AssignFeaturesToGrid()
{
  int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
  for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
    for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
      mGrid[i][j].reserve(nReserve);

  for (int i = 0; i < N; i++)
  {
    const cv::KeyPoint &kp = mvKeysUn[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY))
      mGrid[nGridPosX][nGridPosY].push_back(i);
  }
}

int Frame::detectFeaturesInFrame()
{
  (*OrbExt)(mImGray, cv::Mat(), mvKeys, mDescriptors);

  N = mvKeys.size();
  if (mvKeys.empty())
    return 0;

  UndistortKeyPoints();
  ComputeStereoFromRGBD();

  std::vector<cv::KeyPoint> allFeatures;
  std::vector<cv::KeyPoint> allFeaturesUn;
  std::vector<float> allRight;
  std::vector<float> allDepth;

  for (int i = 0; i < N; ++i)
  {
    if (mvuRight[i] > 0)
    {
      allFeatures.push_back(mvKeys[i]);
      allFeaturesUn.push_back(mvKeysUn[i]);
      allRight.push_back(mvuRight[i]);
      allDepth.push_back(mvDepth[i]);
    }
  }

  mvKeys = allFeatures;
  mvKeysUn = allFeaturesUn;
  mvuRight = allRight;
  mvDepth = allDepth;
  N = allFeatures.size();

  pointsMatches = std::vector<MapPoint *>(N, nullptr);
  outlierFlag = std::vector<bool>(N, false);

  AssignFeaturesToGrid();

  return N;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
  pMP->mbTrackInView = false;

  // 3D in camera coordinates
  Eigen::Vector3d Pc = mTcw.inverse() * pMP->GetWorldPos();
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
  Eigen::Vector3d PO = pMP->GetWorldPos() - mTcw.translation();
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
  pMP->mTrackProjZ = PcZ;
  pMP->mnTrackScaleLevel = nPredictedLevel;
  pMP->mTrackViewCos = viewCos;

  return true;
}

std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
{
  std::vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX = std::max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
  if (nMinCellX >= FRAME_GRID_COLS)
    return vIndices;

  const int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0)
    return vIndices;

  const int nMinCellY = std::max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS)
    return vIndices;

  const int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
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

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
  posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
  posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

  //Keypoint's coordinates are undistorted, which could cause to go out of the image
  if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
    return false;

  return true;
}

void Frame::ComputeBoW()
{
  if (mBowVec.empty())
  {
    std::vector<cv::Mat> vCurrentDesc = ToDescriptorVector(mDescriptors);
    OrbVoc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

void Frame::UndistortKeyPoints()
{
  if (mDistCoef.at<float>(0) == 0.0)
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
  cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
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

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
  if (mDistCoef.at<float>(0) != 0.0)
  {
    cv::Mat mat(4, 2, CV_32F);
    mat.at<float>(0, 0) = 0.0;
    mat.at<float>(0, 1) = 0.0;
    mat.at<float>(1, 0) = imLeft.cols;
    mat.at<float>(1, 1) = 0.0;
    mat.at<float>(2, 0) = 0.0;
    mat.at<float>(2, 1) = imLeft.rows;
    mat.at<float>(3, 0) = imLeft.cols;
    mat.at<float>(3, 1) = imLeft.rows;

    // Undistort corners
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    mnMinX = std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
    mnMaxX = std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
    mnMinY = std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
    mnMaxY = std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));
  }
  else
  {
    mnMinX = 0.0f;
    mnMaxX = imLeft.cols;
    mnMinY = 0.0f;
    mnMaxY = imLeft.rows;
  }
}

void Frame::ComputeStereoFromRGBD()
{
  mvuRight = std::vector<float>(N, -1);
  mvDepth = std::vector<float>(N, -1);

  for (int i = 0; i < N; i++)
  {
    const cv::KeyPoint &kp = mvKeys[i];
    const cv::KeyPoint &kpU = mvKeysUn[i];

    const float &v = kp.pt.y;
    const float &u = kp.pt.x;

    const float d = mImDepth.at<float>(v, u);

    if (d > 0)
    {
      mvDepth[i] = d;
      mvuRight[i] = kpU.pt.x - mbf / d;
    }
  }
}

void Frame::CreateRelocalisationPoints()
{
  if (N == 0)
  {
    detectFeaturesInFrame();
  }

  mvRelocPoints.resize(N);

  for (int i = 0; i < N; ++i)
  {
    if (mvuRight[i] > 0)
    {
      float z = mvDepth[i];
      Eigen::Vector3d PtC;
      cv::KeyPoint &Key = mvKeysUn[i];
      PtC(0) = z * invfx * (Key.pt.x - cx);
      PtC(1) = z * invfy * (Key.pt.y - cy);
      PtC(2) = z;
      mvRelocPoints[i] = PtC;
    }
  }
}

} // namespace slam