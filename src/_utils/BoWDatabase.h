#ifndef _BOW_DATA_BASE_H
#define _BOW_DATA_BASE_H

#include <ORBVocabulary.h>
#include <mutex>
#include <vector>
#include <list>
#include <set>

namespace slam
{

class KeyFrame;
class Frame;

class BoWDatabase
{
public:
    BoWDatabase(ORBVocabulary *voc);
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

#endif