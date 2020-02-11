#pragma once
#include <mutex>
#include <ORBextractor.h>
#include "Map.h"
#include "Frame.h"
#include "MapPoint.h"
// #include "ORBVocabulary.h"

namespace SLAM
{

class Map;
class Frame;
class MapPoint;

class KeyFrame
{
public:
    KeyFrame(Frame *F, Map *map, ORB_SLAM2::ORBextractor *pExtractor);

private:
    void UndistortKeys();
    void AssignFeaturesToGrid();
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
    void ComputeDepth(const cv::Mat depth);

public:
    KeyFrame(Frame &F, Map *pMap);

    // Bag of Words Representation
    void ComputeBoW();
    void SetBadFlag();

    bool IsInFrustum(MapPoint *pMP, float viewingCosLimit);
    void AddMapPoint(MapPoint *pMP, const size_t &idx);

    // Covisibility Graph
    void UpdateConnections();
    void UpdateBestCovisibles();
    void AddConnection(KeyFrame *pKF, int &weight);
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
    std::vector<size_t> GetFeaturesInArea(float &x, float &y, float &r, int minLevel, int maxLevel);

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
    int mnGridCols;
    int mnGridRows;
    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> mGrid;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Calibration parameters
    float mbf, mThDepth;
    int mnMinX, mnMinY, mnMaxX, mnMaxY;
    float fx, fy, cx, cy, invfx, invfy;

    // Number of KeyPoints
    int N;

    // KeyPoints
    std::vector<float> mvDepth;
    std::vector<float> mvuRight;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    cv::Mat mDescriptors;

    // MapPoints associated to keypoints
    std::vector<bool> mvbOutlier;
    std::vector<MapPoint *> mvpMapPoints;
    std::vector<MapPoint *> mvpObservedMapPoints;

    // BoW
    // DBoW2::BowVector mBowVec;
    // DBoW2::FeatureVector mFeatVec;

    // Feature extractor
    // ORB_SLAM2::ORBVocabulary *mpORBvocabulary;

    // Scale data
    // This is extracted from ORB extractor

    // Total scale levels
    int mnScaleLevels;
    // scale factor of each level
    float mfScaleFactor;
    // log scale factor of each level
    float mfLogScaleFactor;
    // scale pyramid
    std::vector<float> mvScaleFactors;
    // scale pyramid ^2
    std::vector<float> mvLevelSigma2;
    // inverse scale pyramid^2
    std::vector<float> mvInvLevelSigma2;

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

    ORB_SLAM2::ORBextractor *mpExtractor;
};

} // namespace SLAM