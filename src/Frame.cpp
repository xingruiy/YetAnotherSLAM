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
      mImDepth(F.mImDepth)
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

  // ORB extraction
  // ExtractORB(imGray);

  // N = mvKeys.size();
  // mvpMapPoints.resize(N, static_cast<MapPoint *>(NULL));

  // ComputeDepth(imDepth);
}

void Frame::SetPose(const cv::Mat &Tcw)
{
}

void Frame::ExtractORB()
{
  ExtractORB(mImGray);

  N = mvKeys.size();
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

    const float d = imDepth.at<float>(v, u);

    if (d > 0)
    {
      mvDepth[i] = d;
    }
  }
}