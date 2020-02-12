#include "Frame.h"

namespace SLAM
{

// unsigned long Frame::mnNextId = 0;

Frame::~Frame()
{
  // printf("frame %lu released\n", mnId);
}

Frame::Frame(const Frame &F) : mTimeStamp(F.mTimeStamp), mImGray(F.mImGray), mImDepth(F.mImDepth),
                               T_frame2Ref(F.T_frame2Ref), mTcw(F.mTcw)
{
}

Frame::Frame(cv::Mat image, cv::Mat depth, double timeStamp)
    : mTimeStamp(timeStamp)
{
  mImGray = image.clone();
  mImDepth = depth.clone();

  // mnId = mnNextId++;
}

} // namespace SLAM