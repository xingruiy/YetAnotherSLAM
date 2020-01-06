#include "Frame.h"

unsigned long Frame::mnNextId = 0;
bool Frame::mbInitialized = false;
float Frame::mbf, Frame::mThDepth;
int Frame::width, Frame::height;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame(const Frame &F)
    : mpORBvocabulary(F.mpORBvocabulary), mpORBextractor(F.mpORBextractor),
      mTimeStamp(F.mTimeStamp), mTcw(F.mTcw), mnScaleLevels(F.mnScaleLevels),
      mfScaleFactor(F.mfScaleFactor), mfLogScaleFactor(F.mfLogScaleFactor),
      mvScaleFactors(F.mvScaleFactors), mvInvScaleFactors(F.mvInvScaleFactors),
      mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2),
      mvKeys(F.mvKeys), mvpMapPoints(F.mvpMapPoints), mImGray(F.mImGray),
      mImDepth(F.mImDepth), mvbOutlier(F.mvbOutlier)
{
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &ts,
             const Eigen::Matrix3d &K, const float &bf, const float &thDepth,
             ORB_SLAM2::ORBextractor *extractor, ORB_SLAM2::ORBVocabulary *voc)
    : mpORBvocabulary(voc), mpORBextractor(extractor), mTimeStamp(ts)
{
  if (!mbInitialized)
  {
    fx = K(0, 0);
    fy = K(1, 1);
    cx = K(0, 2);
    cy = K(1, 2);
    mbf = bf;
    mThDepth = thDepth;
    invfx = 1.0 / fx;
    invfy = 1.0 / fy;
    width = imGray.cols;
    height = imGray.rows;
    mK = K;
    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(width);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(height);
    mbInitialized = true;
  }

  // Frame ID
  mnId = mnNextId++;

  // Copy images
  mImGray = imGray.clone();
  mImDepth = imDepth.clone();

  // Scale Level Info
  mnScaleLevels = mpORBextractor->GetLevels();
  mfScaleFactor = mpORBextractor->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractor->GetScaleFactors();
  mvInvScaleFactors = mpORBextractor->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractor->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractor->GetInverseScaleSigmaSquares();
}

void Frame::SetPose(const cv::Mat &Tcw)
{
}

void Frame::ExtractORB()
{
  ExtractORB(mImGray);

  AssignFeaturesToGrid();

  N = mvKeys.size();
  mvbOutlier.resize(N, false);
  mvpMapPoints.resize(N, static_cast<MapPoint *>(NULL));

  ComputeDepth(mImDepth);
}

void Frame::ExtractORB(const cv::Mat &imGray)
{
  (*mpORBextractor)(imGray, cv::Mat(), mvKeys, mDescriptors);
}

void Frame::ComputeDepth(const cv::Mat &imDepth)
{
  mvDepth = std::vector<float>(N, -1);
  mvuRight = std::vector<float>(N, -1);

  for (int i = 0; i < N; i++)
  {
    const cv::KeyPoint &kp = mvKeys[i];
    const float &v = kp.pt.y;
    const float &u = kp.pt.x;

    const float d = imDepth.at<float>(cv::Point2f(u, v));

    if (d > 0)
    {
      mvDepth[i] = d;
      mvuRight[i] = kp.pt.x - mbf / d;
    }
  }
}

void Frame::AssignFeaturesToGrid()
{
  int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
  for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
    for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
      mGrid[i][j].reserve(nReserve);

  for (int i = 0; i < N; i++)
  {
    const cv::KeyPoint &kp = mvKeys[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY))
      mGrid[nGridPosX][nGridPosY].push_back(i);
  }
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
  posX = round((kp.pt.x) * mfGridElementWidthInv);
  posY = round((kp.pt.y) * mfGridElementHeightInv);

  //Keypoint's coordinates are undistorted, which could cause to go out of the image
  if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
    return false;

  return true;
}

bool Frame::IsInFrustum(MapPoint *pMP, float viewingCosLimit)
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