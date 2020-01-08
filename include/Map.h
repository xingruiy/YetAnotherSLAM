#pragma once
#include <set>
#include <mutex>
#include <memory>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

class Frame;
class MapPoint;
class KeyFrame;

using namespace std;

class Map
{
public:
    void Reset();
    void AddKeyFrame(KeyFrame *pKF);
    void AddMapPoint(MapPoint *pMP);

    unsigned long KeyFramesInMap();
    std::vector<KeyFrame *> GetAllKeyFrames();
    std::vector<MapPoint *> GetAllMapPoints();

private:
    set<KeyFrame *> mspKeyFrames;
    set<MapPoint *> mspMapPoints;

    std::mutex mMutexMap;
};
