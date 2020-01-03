#include "Frame.h"

unsigned long Frame::mnNextId = 0;
bool Frame::mbInitialized = false;
int Frame::width, Frame::height;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;

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
             const Eigen::Matrix3d &K, ORB_SLAM2::ORBextractor *extractor,
             ORB_SLAM2::ORBVocabulary *voc)
    : mpORBvocabulary(voc), mpORBextractor(extractor), mTimeStamp(ts)
{
  if (!mbInitialized)
  {
    fx = K(0, 0);
    fy = K(1, 1);
    cx = K(0, 2);
    cy = K(1, 2);
    invfx = 1.0 / fx;
    invfy = 1.0 / fy;
    width = imGray.cols;
    height = imGray.rows;
    mK = K;
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

  for (int i = 0; i < N; i++)
  {
    const cv::KeyPoint &kp = mvKeys[i];
    const float &v = kp.pt.y;
    const float &u = kp.pt.x;

    const float d = imDepth.at<float>(cv::Point2f(u, v));

    if (d > 0)
    {
      mvDepth[i] = d;
    }
  }
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
  // cv::Mat Pn = pMP->GetNormal();

  // const float viewCos = PO.dot(Pn) / dist;

  // if (viewCos < viewingCosLimit)
  //   return false;

  // // Predict scale in the image
  // const int nPredictedLevel = pMP->PredictScale(dist, this);

  // // Data used by the tracking
  pMP->mbTrackInView = true;
  // pMP->mTrackProjX = u;
  // pMP->mTrackProjXR = u - mbf * invz;
  // pMP->mTrackProjY = v;
  // pMP->mnTrackScaleLevel = nPredictedLevel;
  // pMP->mTrackViewCos = viewCos;

  return true;
}