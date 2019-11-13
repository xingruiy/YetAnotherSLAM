#include "dataStruct/keyFrame.h"

size_t KeyFrame::nextKFId = 0;

KeyFrame::KeyFrame(const Frame &F)
    : keyPoints(F.cvKeyPoints),
      mapPoints(F.mapPoints),
      descriptors(F.descriptors),
      KFId(nextKFId++)
{
}