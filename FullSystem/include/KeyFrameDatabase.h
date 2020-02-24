#pragma once
#include <ORBVocabulary.h>
#include <vector>
#include <list>
#include <set>

#include "KeyFrame.h"
#include "Frame.h"
#include <mutex>

namespace SLAM
{

class KeyFrame;
class Frame;

class KeyFrameDatabase
{
public:
    KeyFrameDatabase(const ORB_SLAM2::ORBVocabulary &voc);

    void add(KeyFrame *pKF);

    void erase(KeyFrame *pKF);

    void clear();

    // Loop Detection
    std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame *pKF, float minScore);

    // Relocalization
    std::vector<KeyFrame *> DetectRelocalizationCandidates(Frame *F);

protected:
    // Associated vocabulary
    const ORB_SLAM2::ORBVocabulary *mpVoc;

    // Inverted file
    std::vector<std::list<KeyFrame *>> mvInvertedFile;

    // Mutex
    std::mutex mMutex;
};

} // namespace SLAM
