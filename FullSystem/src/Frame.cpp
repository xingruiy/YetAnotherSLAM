#include "Frame.h"
#include "Converter.h"

namespace SLAM
{

unsigned long Frame::mnNextId = 0;

Frame::~Frame()
{
}

Frame::Frame(const Frame &F) : mTimeStamp(F.mTimeStamp), mImGray(F.mImGray), mImDepth(F.mImDepth),
                               mRelativePose(F.mRelativePose), mTcw(F.mTcw)
{
  mnId = mnNextId++;
}

Frame::Frame(cv::Mat image, cv::Mat depth, double timeStamp, ORBextractor *pExtractor)
    : mTimeStamp(timeStamp), mpORBExtractor(pExtractor)
{
  mImGray = image.clone();
  mImDepth = depth.clone();
}

void Frame::ExtractORBFeatures()
{
  (*mpORBExtractor)(mImGray, cv::Mat(), mvKeys, mDescriptors);
}

void Frame::ComputeBoW(ORB_SLAM2::ORBVocabulary *voc)
{
  if (mBowVec.empty())
  {
    std::vector<cv::Mat> vCurrentDesc = ORB_SLAM2::Converter::toDescriptorVector(mDescriptors);
    voc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

} // namespace SLAM