#include "Frame.h"

namespace SLAM
{

unsigned long Frame::mnNextId = 0;
bool Frame::mbInitialized = false;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame(const Frame &F)
    : mpORBvocabulary(F.mpORBvocabulary), mpORBextractor(F.mpORBextractor),
      mTimeStamp(F.mTimeStamp), mTcw(F.mTcw), mnScaleLevels(F.mnScaleLevels),
      mfScaleFactor(F.mfScaleFactor), mfLogScaleFactor(F.mfLogScaleFactor),
      mvScaleFactors(F.mvScaleFactors), mvInvScaleFactors(F.mvInvScaleFactors),
      mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2),
      mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mvpMapPoints(F.mvpMapPoints),
      mImGray(F.mImGray), mImDepth(F.mImDepth), mvbOutlier(F.mvbOutlier),
      mbf(F.mbf), mThDepth(F.mThDepth)
{
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &ts,
             ORB_SLAM2::ORBextractor *extractor, ORB_SLAM2::ORBVocabulary *voc)
    : mTimeStamp(ts), mDistCoef(g_distCoeff.clone()), mbf(g_bf),
      mThDepth(g_thDepth), mpORBvocabulary(voc), mpORBextractor(extractor)
{
  cv::Mat cvK = cv::Mat::eye(3, 3, CV_32F);
  cvK.at<float>(0, 0) = g_fx[0];
  cvK.at<float>(1, 1) = g_fy[0];
  cvK.at<float>(0, 2) = g_cx[0];
  cvK.at<float>(1, 2) = g_cy[0];
  cvK.copyTo(mK);

  if (!mbInitialized)
  {
    ComputeImageBounds(imGray);

    fx = g_fx[0];
    fy = g_fy[0];
    cx = g_cx[0];
    cy = g_cy[0];
    invfx = 1.0 / fx;
    invfy = 1.0 / fy;

    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

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
  // For predicting scales
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractor->GetScaleFactors();
  mvInvScaleFactors = mpORBextractor->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractor->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractor->GetInverseScaleSigmaSquares();
}

void Frame::ExtractORB()
{
  // Extract ORB features
  ExtractORB(mImGray);

  N = mvKeys.size();
  mvbOutlier.resize(N, false);
  mvpMapPoints.resize(N, NULL);

  // Generate undistorted key points
  UndistortKeyPoints();

  // Assign all key points to the grid
  AssignFeaturesToGrid();

  // Get depth from raw input
  ComputeDepth(mImDepth);
}

void Frame::ExtractORB(const cv::Mat &imGray)
{
  (*mpORBextractor)(imGray, cv::Mat(), mvKeys, mDescriptors);
}

void Frame::ComputeDepth(const cv::Mat &depth_image)
{
  // The virtual right coordinate
  mvuRight = vector<float>(N, -1);
  // Key point depth
  mvDepth = vector<float>(N, -1);

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
      mvuRight[i] = kpU.pt.x - mbf / d;
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
    const cv::KeyPoint &kp = mvKeysUn[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY))
      mGrid[nGridPosX][nGridPosY].push_back(i);
  }
}

void Frame::UndistortKeyPoints()
{
  if (N == 0)
    return;

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

void Frame::ComputeImageBounds(const cv::Mat &img)
{
  if (mDistCoef.at<float>(0) != 0.0)
  {
    cv::Mat mat(4, 2, CV_32F);
    mat.at<float>(0, 0) = 0.0;
    mat.at<float>(0, 1) = 0.0;
    mat.at<float>(1, 0) = img.cols;
    mat.at<float>(1, 1) = 0.0;
    mat.at<float>(2, 0) = 0.0;
    mat.at<float>(2, 1) = img.rows;
    mat.at<float>(3, 0) = img.cols;
    mat.at<float>(3, 1) = img.rows;

    // Undistort corners
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
    mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
    mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
    mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
  }
  else
  {
    mnMinX = 0.0f;
    mnMaxX = img.cols;
    mnMinY = 0.0f;
    mnMaxY = img.rows;
  }
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

  if (u < mnMinX || u > mnMaxX)
    return false;
  if (v < mnMinY || v > mnMaxY)
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

} // namespace SLAM