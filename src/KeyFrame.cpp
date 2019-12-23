#include "KeyFrame.h"

unsigned long KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(const Frame &F, Map *pMap)
    : mpMap(pMap), mvKeys(F.mvKeys), mTcw(F.mTcw), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
      mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
      mvInvLevelSigma2(F.mvInvLevelSigma2), N(F.N), fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx),
      invfy(F.invfy), mvpMapPoints(F.mvpMapPoints), mvDepth(F.mvDepth)
{
  mnId = nNextId++;
}