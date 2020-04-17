#ifndef _LOCAL_MAPPING_H
#define _LOCAL_MAPPING_H

#include <memory>
#include <mutex>
#include <ORBVocabulary.h>

#include "Map.h"
#include "GlobalDef.h"
#include "KeyFrame.h"
#include "Viewer.h"
#include "ORBextractor.h"
#include "LoopClosing.h"

namespace slam
{

class Viewer;
class Map;
class LoopClosing;

class LocalMapping
{
public:
    LocalMapping(ORBVocabulary *pVoc, Map *mpMap);
    void SetLoopCloser(LoopClosing *pLoopCloser);

    void Run();
    void InsertKeyFrame(KeyFrame *pKF);

    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue()
    {
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

private:
    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();
    void MapPointCulling();
    void SearchInNeighbors();
    void KeyFrameCulling();
    void ResetIfRequested();

    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map *mpMap;
    KeyFrame *currKF;

    LoopClosing *mpLoopCloser;

    std::list<KeyFrame *> mlNewKeyFrames;
    std::list<MapPoint *> mlpRecentAddedMapPoints;
    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;
};

} // namespace slam

#endif