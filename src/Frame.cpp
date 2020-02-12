#include "Frame.h"

namespace SLAM
{

unsigned long Frame::mnNextId = 0;

Frame::~Frame()
{
  // printf("frame %lu released\n", mnId);
}

Frame::Frame(cv::Mat image, cv::Mat depth, double timeStamp)
    : mTimeStamp(timeStamp), mbIsKeyFrame(false)
{
  mImGray = image.clone();
  mImDepth = depth.clone();

  mnId = mnNextId++;
}

} // namespace SLAM