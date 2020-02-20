#pragma once
#include <set>
#include <mutex>
#include <memory>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

namespace SLAM
{

class Frame;
class MapPoint;
class KeyFrame;

class Map
{
public:
    void reset();
    void AddKeyFrame(KeyFrame *pKF);
    void AddMapPoint(MapPoint *pMP);
    void EraseKeyFrame(KeyFrame *pKF);
    void EraseMapPoint(MapPoint *pMP);
    void SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs);
    std::vector<MapPoint *> GetReferenceMapPoints();

    std::vector<KeyFrame *> GetAllKeyFrames();
    std::vector<MapPoint *> GetAllMapPoints();

    std::mutex mMutexMapUpdate;

private:
    std::mutex mMutexMap;
    std::set<KeyFrame *> mspKeyFrames;
    std::set<MapPoint *> mspMapPoints;
    std::vector<MapPoint *> mvpReferenceMapPoints;
};

} // namespace SLAM