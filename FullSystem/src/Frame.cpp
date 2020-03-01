#include "Frame.h"
#include "Converter.h"

namespace SLAM
{

long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{
}

Frame::Frame(const Frame &F)
    : mTimeStamp(F.mTimeStamp), mImGray(F.mImGray), mImDepth(F.mImDepth),
      mRelativePose(F.mRelativePose), mTcw(F.mTcw)
{
  mnId = nNextId++;
}

Frame::Frame(const cv::Mat imGray, const cv::Mat imDepth, const double &timeStamp, ORBextractor *extractor)
    : mTimeStamp(timeStamp), mpORBextractor(extractor)
{
  mImGray = imGray.clone();
  mImDepth = imDepth.clone();
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

void Frame::ExtractORB()
{
  (*mpORBextractor)(mImGray, cv::Mat(), mvKeys, mDescriptors);
}

void Frame::SetPose(const Sophus::SE3d &Tcw)
{
  mTcw = Tcw;
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
    std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
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

void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
  mvuRight = std::vector<float>(N, -1);
  mvDepth = std::vector<float>(N, -1);

  for (int i = 0; i < N; i++)
  {
    const cv::KeyPoint &kp = mvKeys[i];
    const cv::KeyPoint &kpU = mvKeysUn[i];

    const float &v = kp.pt.y;
    const float &u = kp.pt.x;

    const float d = imDepth.at<float>(v, u);

    if (d > 0)
    {
      mvDepth[i] = d;
      mvuRight[i] = kpU.pt.x - mbf / d;
    }
  }
}

} // namespace SLAM