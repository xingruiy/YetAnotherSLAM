#pragma once
#include <ORBVocabulary.h>
#include <vector>
#include <list>
#include <set>

#include "KeyFrame.h"
#include "Frame.h"
#include <mutex>

namespace slam
{

class KeyFrame;
class Frame;

class KeyFrameDatabase
{
public:
    KeyFrameDatabase(const ORBVocabulary &voc);

    void add(KeyFrame *pKF);

    void erase(KeyFrame *pKF);

    void clear();

    // Loop Detection
    std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame *pKF, float minScore);

    // Relocalization
    std::vector<KeyFrame *> DetectRelocalizationCandidates(Frame *F);

protected:
    // Associated vocabulary
    const ORBVocabulary *mpVoc;

    // Inverted file
    std::vector<std::list<KeyFrame *>> mvInvertedFile;

    // Mutex
    std::mutex mMutex;
};

} // namespace slam
