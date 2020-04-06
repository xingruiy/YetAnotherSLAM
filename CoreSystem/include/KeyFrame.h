#pragma once
#include <mutex>
#include <ORBextractor.h>
#include <ORBVocabulary.h>
#include <DBoW2/DBoW2/BowVector.h>
#include <DBoW2/DBoW2/FeatureVector.h>
#include "Map.h"
#include "Frame.h"
#include "VoxelMap.h"
#include "MapPoint.h"
#include "KeyFrameDatabase.h"

namespace slam
{

class Map;
class Frame;
class MapPoint;
class KeyFrameDatabase;

class KeyFrame
{
public:
    KeyFrame(const Frame &F, Map *pMap, KeyFrameDatabase *pKFDB);

    // Pose functions
    void SetPose(const Sophus::SE3d &Tcw);
    Sophus::SE3d GetPose();
    Sophus::SE3d GetPoseInverse();
    Eigen::Matrix3d GetRotation();
    Eigen::Vector3d GetTranslation();

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility Graph functions
    int GetWeight(KeyFrame *pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    void AddConnection(KeyFrame *pKF, int &weight);
    void EraseConnection(KeyFrame *pKF);
    std::set<KeyFrame *> GetConnectedKeyFrames();
    std::vector<KeyFrame *> GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame *> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<KeyFrame *> GetCovisiblesByWeight(const int &w);

    // Spanning tree functions
    KeyFrame *GetParent();
    void AddChild(KeyFrame *pKF);
    void EraseChild(KeyFrame *pKF);
    void ChangeParent(KeyFrame *pKF);
    std::set<KeyFrame *> GetChilds();
    bool hasChild(KeyFrame *pKF);

    // Loop Edges
    void AddLoopEdge(KeyFrame *pKF);
    std::set<KeyFrame *> GetLoopEdges();

    // MapPoint observation functions
    void AddMapPoint(MapPoint *pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapPoint *pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP);
    std::set<MapPoint *> GetMapPoints();
    std::vector<MapPoint *> GetMapPointMatches();
    MapPoint *GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, float r, int minlvl = -1, int maxlvl = -1);
    Eigen::Vector3d UnprojectStereo(int i);
    bool UnprojectStereo(Eigen::Vector3d &posWorld, const int &i);

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    bool IsInFrustum(MapPoint *pMP, float viewingCosLimit);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;

    // Variables used by loop closing
    Sophus::SE3d mTcwGBA;
    Sophus::SE3d mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth;  // negative value for monocular points
    const cv::Mat mDescriptors;

    // BoW.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent
    Sophus::SE3d mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;

    // The following variables need to be accessed trough a mutex to be thread safe.
protected:
    // Keyframe in World coord
    Sophus::SE3d mTcw;

    // MapPoints associated to keypoints
    std::vector<MapPoint *> mvpMapPoints;

    // BoW
    KeyFrameDatabase *KFDatabase;
    ORBVocabulary *mpORBvocabulary;

    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> mGrid;

    std::map<KeyFrame *, int> mConnectedKeyFrameWeights;
    std::vector<KeyFrame *> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFrame *mpParent;
    std::set<KeyFrame *> mspChildrens;
    std::set<KeyFrame *> mspLoopEdges;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;

    // The following variables are newly added
public:
    Map *mpMap;

    long unsigned int GetMapId();

    // The original image for visualization
    cv::Mat mImg;
    cv::Mat mDepth;

    // Dense Maps contained
    MapStruct *mpVoxelStruct;
};

} // namespace slam