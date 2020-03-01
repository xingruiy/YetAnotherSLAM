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

namespace SLAM
{

class Map;
class Frame;
class MapPoint;

class KeyFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KeyFrame(const Frame &F, Map *pMap);

    // Pose functions
    void SetPose(const Sophus::SE3d &Tcw);
    Sophus::SE3d GetPose();
    Sophus::SE3d GetPoseInverse();
    Eigen::Matrix3d GetRotation();
    Eigen::Vector3d GetTranslation();

    // Bag of Words Representation
    void ComputeBoW(ORBVocabulary *voc);

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
    Eigen::Vector3d UnprojectKeyPoint(int i);
    bool UnprojectKeyPoint(Eigen::Vector3d &posWorld, const int &i);

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    bool IsInFrustum(MapPoint *pMP, float viewingCosLimit);

public:
    unsigned long mnId;
    static unsigned long nNextId;

    const double mTimeStamp;

    // Variables used by the local mpLocalMapper
    unsigned long mnBALocalForKF;
    unsigned long mnBAFixedForKF;
    unsigned long mnFuseTargetForKF;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;

    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> mGrid;

    // Variables used by loop closing
    Sophus::SE3d mTcwGBA;
    Sophus::SE3d mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;
    cv::Mat mK;

    // Number of KeyPoints
    int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;
    cv::Mat mDescriptors;

    // BoW.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent
    Sophus::SE3d mRelativePose;

    // MapPoints associated to keypoints
    std::vector<bool> mvbOutlier;
    std::vector<MapPoint *> mvpMapPoints;

    // ORB scale pyramid info
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

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
    std::mutex poseMutex;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
    KeyFrame *mReferenceKeyFrame;

    void UndistortKeys();
    void AssignFeaturesToGrid();
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
    void ComputeStereoRGBD(const cv::Mat depth);

    MapStruct *mpVoxelStruct;
    bool mbVoxelStructMarginalized;

    ORBextractor *mpExtractor;
    Map *mpMap;
    cv::Mat mImg;
    bool mbBad;
    bool mbNotErase;
    bool mbToBeErased;
};

} // namespace SLAM