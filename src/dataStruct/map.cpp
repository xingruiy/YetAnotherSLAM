#include "dataStruct/map.h"
#include <fstream>

Map::Map()
{
}

void Map::clear()
{
    std::unique_lock<std::mutex> lock(mapMutex);
    std::unique_lock<std::mutex> lock2(histMutex);
    std::unique_lock<std::mutex> lock3(mutexloop);
    std::unique_lock<std::mutex> lock4(mutexkf);

    currentKeyframe = NULL;
    descriptorDB.release();
    keyFrameDB.clear();
    mapPointDB.clear();
    framePoseOptimized.clear();
    keyframePoseRawAll.clear();
    framePoseRawAll.clear();
    unprocessedKeyframeQueue = std::queue<std::shared_ptr<Frame>>();
    loopClosingKeyframeQueue = std::queue<std::shared_ptr<Frame>>();
}

void Map::addKeyFrame(std::shared_ptr<Frame> kf)
{
    std::unique_lock<std::mutex> lock(mapMutex);
    keyFrameDB.push_back(kf);
    std::unique_lock<std::mutex> lock2(histMutex);
}

std::shared_ptr<Frame> Map::getCurrentKeyframe() const
{
    return currentKeyframe;
}

void Map::setCurrentKeyframe(std::shared_ptr<Frame> kf)
{
    currentKeyframe = kf;
}

void Map::addMapPoint(std::shared_ptr<MapPoint> pt)
{
    std::unique_lock<std::mutex> lock(mapMutex);
    descriptorDB.push_back(pt->getDescriptor());
    mapPointDB.push_back(pt);
}

void Map::addUnprocessedKeyframe(std::shared_ptr<Frame> kf)
{
    std::unique_lock<std::mutex> lock(mutexkf);
    unprocessedKeyframeQueue.push(kf);
}

std::shared_ptr<Frame> Map::getUnprocessedKeyframe()
{
    std::unique_lock<std::mutex> lock(mutexkf);
    std::shared_ptr<Frame> kf = NULL;
    if (!unprocessedKeyframeQueue.empty())
    {
        kf = unprocessedKeyframeQueue.front();
        unprocessedKeyframeQueue.pop();
    }

    return kf;
}

void Map::addLoopClosingKeyframe(std::shared_ptr<Frame> kf)
{
    std::unique_lock<std::mutex> lock(mutexloop);
    loopClosingKeyframeQueue.push(kf);
}

std::shared_ptr<Frame> Map::getLoopClosingKeyframe()
{
    std::unique_lock<std::mutex> lock(mutexloop);
    std::shared_ptr<Frame> kf = NULL;
    if (!loopClosingKeyframeQueue.empty())
    {
        kf = loopClosingKeyframeQueue.front();
        loopClosingKeyframeQueue.pop();
    }

    return kf;
}

std::vector<SE3> Map::getKeyframePoseRaw()
{
    std::unique_lock<std::mutex> lock(histMutex);
    return keyframePoseRawAll;
}

std::vector<SE3> Map::getKeyframePoseOptimized()
{
    std::unique_lock<std::mutex> lock(mapMutex);
    std::vector<SE3> output;
    for (auto kf : keyFrameDB)
        output.push_back(kf->getPoseInGlobalMap());

    return output;
}

std::vector<SE3> Map::getFramePoseRaw()
{
    std::unique_lock<std::mutex> lock(histMutex);
    return framePoseRawAll;
}

std::vector<SE3> Map::getFramePoseOptimized()
{
    std::unique_lock<std::mutex> lock(histMutex);
    std::unique_lock<std::mutex> lock2(mapMutex);
    std::vector<SE3> output;
    for (auto frame : framePoseOptimized)
    {
        auto &kf = frame.second;
        output.push_back(kf->getPoseInGlobalMap() * frame.first);
    }

    return output;
}

std::vector<Vec3f> Map::getMapPointVec3All()
{
    std::vector<Vec3f> pts;
    std::unique_lock<std::mutex> lock(mapMutex);

    for (auto pt : mapPointDB)
        if (pt && !pt->isBad())
            pts.push_back(pt->getPosWorld().cast<float>());

    return pts;
}

void Map::addFramePoseRaw(const SE3 &T)
{
    std::unique_lock<std::mutex> lock(histMutex);
    framePoseRawAll.push_back(T);
}

void Map::addKeyframePoseRaw(const SE3 &T)
{
    std::unique_lock<std::mutex> lock(histMutex);
    keyframePoseRawAll.push_back(T);
}

void Map::addFramePose(const SE3 &T, std::shared_ptr<Frame> kf)
{
    std::unique_lock<std::mutex> lock(histMutex);
    framePoseOptimized.push_back(std::make_pair(T, kf));
}

void Map::writeToDisk(const char *fileName)
{
    std::ofstream file(fileName);
    if (file.is_open())
    {
    }

    file.close();
}

void Map::readFromDisk(const char *fileName)
{
    std::ifstream file(fileName);
    if (file.is_open())
    {
    }

    file.close();
}