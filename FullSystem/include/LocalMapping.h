/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include <memory>
#include <mutex>
#include <ORBextractor.h>
#include <ORBVocabulary.h>

#include "Map.h"
#include "GlobalDef.h"
#include "KeyFrame.h"
#include "Viewer.h"
#include "LoopClosing.h"

namespace SLAM
{

class Viewer;
class LoopClosing;

class LocalMapping
{
public:
    LocalMapping(ORBVocabulary *pVoc, Map *pMap);

    void setLoopCloser(LoopClosing *pLoopCloser);

    void setViewer(Viewer *pViewer);

    // Main function
    void Run();

    void AddKeyFrameCandidate(KeyFrame *pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

private:
    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    // Local mapping, moved form tracker in ORB_SLAM2
    void UpdateLocalMap();
    void UpdateKeyFrame();
    int MatchLocalPoints();

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    // keyframe candidate
    std::mutex mMutexNewKFs;
    std::list<KeyFrame *> mlNewKeyFrames;
    KeyFrame *mpCurrentKeyFrame;
    KeyFrame *mpLastKeyFrame;
    KeyFrame *mpReferenceKeyframe;

    ORBVocabulary *ORBvocabulary;

    std::vector<KeyFrame *> mvpLocalKeyFrames;
    std::vector<MapPoint *> mvpLocalMapPoints;

    Map *mpMap;
    Viewer *mpViewer;
    LoopClosing *mpLoopCloser;
    std::list<MapPoint *> mlpRecentAddedMapPoints;

    cv::Mat mImg;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;
};

} // namespace SLAM

#endif