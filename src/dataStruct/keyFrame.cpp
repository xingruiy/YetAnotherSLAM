#include "dataStruct/keyFrame.h"

size_t KeyFrame::nextKFId = 0;

KeyFrame::KeyFrame(const Frame &F)
    : KFId(nextKFId++), imDepth(F.imDepth), imRGB(F.imRGB)
{
}

double KeyFrame::getDepth(double x, double y)
{
  return (double)imDepth.ptr<float>((int)round(y))[(int)round(x)];
}

std::vector<size_t> KeyFrame::getKeyPointsInArea(const double x, const double y, const double th)
{
  std::vector<size_t> indices;
  for (int i = 0; i < keyPoints.size(); ++i)
  {
    const auto &kp = keyPoints[i];
    Eigen::Vector2d dist(x - kp.pt.x, y - kp.pt.y);
    if (dist.norm() < th)
    {
      indices.push_back(i);
    }
  }

  return indices;
}