#include "KeyFrame.h"
#include "Converter.h"

unsigned long KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(const Frame &F, Map *pMap)
    : mpMap(pMap), mvKeys(F.mvKeys), mTcw(F.mTcw), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
      mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
      mvInvLevelSigma2(F.mvInvLevelSigma2), N(F.N), fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx),
      invfy(F.invfy), mvpMapPoints(F.mvpMapPoints), mvDepth(F.mvDepth), mvbOutlier(F.mvbOutlier)
{
  mnId = nNextId++;
}

void KeyFrame::ComputeBoW()
{
  if (mBowVec.empty() || mFeatVec.empty())
  {
    vector<cv::Mat> vCurrentDesc = ORB_SLAM2::Converter::toDescriptorVector(mDescriptors);
    // Feature vector associate features with nodes in the 4th level (from leaves up)
    // We assume the vocabulary tree has 6 levels, change the 4 otherwise
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}
