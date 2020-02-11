#pragma once
#include <mutex>
#include "Map.h"
#include "Frame.h"
#include "MapPoint.h"
#include "ORBVocabulary.h"

namespace SLAM
{

class Map;
class Frame;
class MapPoint;

class KeyFrame
{
public:
    KeyFrame(const Frame &F, Map *pMap);

    // Bag of Words Representation
    void ComputeBoW();
    void SetBadFlag();

    bool IsInFrustum(MapPoint *pMP, float viewingCosLimit);
    void AddMapPoint(MapPoint *pMP, const size_t &idx);

    // Covisibility Graph
    void UpdateConnections();
    void UpdateBestCovisibles();
    void AddConnection(KeyFrame *pKF, const int &weight);
    void EraseConnection(KeyFrame *pKF);
    int GetWeight(KeyFrame *pKF);

    // Spanning tree functions
    void AddChild(KeyFrame *pKF);
    void EraseChild(KeyFrame *pKF);
    void ChangeParent(KeyFrame *pKF);
    std::set<KeyFrame *> GetChilds();
    KeyFrame *GetParent();
    bool hasChild(KeyFrame *pKF);
    std::vector<KeyFrame *> GetVectorCovisibleKeyFrames();

    Eigen::Vector3d UnprojectKeyPoint(int i);
    std::vector<MapPoint *> GetMapPointMatches();
    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const;

public:
    bool isBad();
    bool mbBad;

    bool mbNotErase;
    bool mbToBeErased;

    cv::Mat mImGray;
    double mTimeStamp;

    unsigned long mnId;
    static unsigned long nNextId;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> orbGrid;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Calibration parameters
    const float mbf, mThDepth;
    const int mnMinX, mnMinY, mnMaxX, mnMaxY;
    const float fx, fy, cx, cy, invfx, invfy;

    // Number of KeyPoints
    const int N;

    // KeyPoints
    const std::vector<float> mvDepth;
    const std::vector<float> mvuRight;
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const cv::Mat mDescriptors;

    // MapPoints associated to keypoints
    std::vector<bool> mvbOutlier;
    std::vector<MapPoint *> mvpMapPoints;
    std::vector<MapPoint *> mvpObservedMapPoints;

    // BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Feature extractor
    ORB_SLAM2::ORBVocabulary *mpORBvocabulary;

    // Scale data
    // This is extracted from ORB extractor

    // Total scale levels
    const int mnScaleLevels;
    // scale factor of each level
    const float mfScaleFactor;
    // log scale factor of each level
    const float mfLogScaleFactor;
    // scale pyramid
    const std::vector<float> mvScaleFactors;
    // scale pyramid ^2
    const std::vector<float> mvLevelSigma2;
    // inverse scale pyramid^2
    const std::vector<float> mvInvLevelSigma2;

    Map *mpMap;

    // Covisiblity graph
    std::map<KeyFrame *, int> mConnectedKeyFrameWeights;
    std::vector<KeyFrame *> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFrame *mpParent;
    std::set<KeyFrame *> mspChildrens;
    std::set<KeyFrame *> mspLoopEdges;

    // Keyframe in World coord
    Sophus::SE3d mTcw;
    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

} // namespace SLAM