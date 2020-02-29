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
    bool IsInFrustum(MapPoint *pMP, float viewingCosLimit);
    void AddMapPoint(MapPoint *pMP, const size_t &idx);

    cv::Mat GetRotation() const;
    cv::Mat GetTranslation() const;
    cv::Mat GetInvTransform() const;

    bool UnprojectKeyPoint(Eigen::Vector3d &posWorld, const int &i);
    Eigen::Vector3d UnprojectKeyPoint(int i);

    MapPoint *GetMapPoint(const size_t &idx);
    std::set<MapPoint *> GetMapPoints();
    std::vector<MapPoint *> GetMapPointMatches();
    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, float r, int minlvl = -1, int maxlvl = -1);
    bool IsInImage(const float &x, const float &y) const;
    bool isBad();

    // Bag of Words Representation
    void ComputeBoW(ORB_SLAM2::ORBVocabulary *voc);
    // Destroy frame
    void SetBadFlag();
    void EraseMapPointMatch(MapPoint *pMP);
    void EraseMapPointMatch(const size_t &idx);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP);

    // Covisibility Graph
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

    void SetPose(cv::Mat Tcw);

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

public:
    Map *mpMap;
    cv::Mat mImg;
    bool mbBad;
    bool mbNotErase;
    bool mbToBeErased;

    double mTimeStamp;

    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> mGrid;

    // Calibration parameters
    float mb, mbf, mThDepth;
    float fx, fy, cx, cy, invfx, invfy;
    cv::Mat mK;

    // KeyPoints configurations
    int N; // Number of KeyPoints
    std::vector<float> mvDepth;
    std::vector<float> mvuRight;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    cv::Mat mDescriptors;

    // MapPoints associated to keypoints
    std::vector<bool> mvbOutlier;
    std::vector<MapPoint *> mvpMapPoints;

    // ORB scale pyramid info
    int mnScaleLevels;                   // Total scale levels
    float mfScaleFactor;                 // scale factor of each level
    float mfLogScaleFactor;              // log scale factor of each level
    std::vector<float> mvScaleFactors;   // scale pyramid
    std::vector<float> mvLevelSigma2;    // scale pyramid ^2
    std::vector<float> mvInvLevelSigma2; // inverse scale pyramid^2

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
    Sophus::SE3d mRelativePose;
    std::mutex poseMutex;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
    KeyFrame *mReferenceKeyFrame;

    // Variables used by the local mpLocalMapper
    unsigned long mnBALocalForKF;
    unsigned long mnBAFixedForKF;
    unsigned long mnFuseTargetForKF;

    unsigned long mnId;
    static unsigned long nNextId;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    void UndistortKeys();
    void AssignFeaturesToGrid();
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
    void ComputeStereoRGBD(const cv::Mat depth);

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;

    MapStruct *mpVoxelStruct;
    bool mbVoxelStructMarginalized;

    ORBextractor *mpExtractor;
};

} // namespace SLAM