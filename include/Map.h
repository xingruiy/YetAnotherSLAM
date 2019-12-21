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
    // public:
    //     Map();

    //     void clear();

    //     // TODO: not yet implemented
    //     void writeToDisk(const char *fileName);

    //     // TODO: not yet implemented
    //     void readFromDisk(const char *fileName);

    //     // Insert new key frame into the map
    //     void addKeyFrame(std::shared_ptr<KeyFrame> KF);

    //     // Insert new map point into the map
    //     void addMapPoint(std::shared_ptr<MapPoint> MP);

    //     // Get map ponit pos
    //     void getMapPoint(std::vector<Eigen::Vector3f> &MPs);

    // public:
    //     Mat descriptorDB;
    //     std::vector<std::shared_ptr<MapPoint>> mapPointDB;
    //     std::vector<std::shared_ptr<KeyFrame>> keyFrameDB;

    //     std::mutex mapMutex;
    //     std::shared_ptr<Frame> currentKeyframe;

    ////////////////////////////
public:
    void AddKeyFrame(KeyFrame *pKF);
    void AddMapPoint(MapPoint *pMP);

    vector<MapPoint *> GetMapPointVec();

private:
    set<KeyFrame *> mspKeyFrames;
    set<MapPoint *> mspMapPoints;

    std::mutex mMutexMap;
};
