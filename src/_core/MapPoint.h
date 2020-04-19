#ifndef _MAP_POINT_H
#define _MAP_POINT_H

#include <mutex>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace slam
{

class Map;
class Frame;
class KeyFrame;

struct PatchDescriptor
{
    float data[25];

    inline float getScoreNCC(PatchDescriptor &other)
    {
    }
};

class MapPoint
{
public:
    MapPoint(const Eigen::Vector3d &Pos, KeyFrame *pRefKF, Map *mpMap);
    MapPoint(const Eigen::Vector3d &pos, Map *mpMap, KeyFrame *pRefKF, const int &idxF);

    void SetWorldPos(const Eigen::Vector3d &pos);
    Eigen::Vector3d GetWorldPos();

    // Get Normal
    Eigen::Vector3d GetNormal();
    KeyFrame *GetReferenceKeyFrame();

    std::map<KeyFrame *, size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame *pKF, size_t idx);
    void EraseObservation(KeyFrame *pKF);

    int GetIndexInKeyFrame(KeyFrame *pKF);
    bool IsInKeyFrame(KeyFrame *pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint *pMP);
    MapPoint *GetReplaced();

    void IncreaseFound(int n = 1);
    void IncreaseVisible(int n = 1);
    float GetFoundRatio();

    void ComputeDistinctiveDescriptors();
    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, Frame *pFrame);
    int PredictScale(const float &currentDist, KeyFrame *pKF);

public:
    int mnId;
    static int nNextId;
    long int mnFirstKFid;
    int mnLastFrameSeen;
    int nObs;
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjZ;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    int mnTrackReferenceForFrame;
    int mnBALocalForKF;
    int mnFuseCandidateForKF;
    int mnLoopPointForKF;
    int mnCorrectedByKF;
    int mnCorrectedReference;
    Eigen::Vector3d mPosGBA;
    int mnBAGlobalForKF;
    static std::mutex mGlobalMutex;

public:
    Eigen::Vector3d mWorldPos;
    std::map<KeyFrame *, size_t> mObservations;
    Eigen::Vector3d mAvgViewingDir;
    cv::Mat mDescriptor;
    KeyFrame *mpRefKF;
    int mnVisible;
    int mnFound;
    bool mbBad;
    MapPoint *mpReplaced;
    float mfMinDistance;
    float mfMaxDistance;
    Map *mpMap;
    std::mutex mMutexPos;
    std::mutex mMutexFeatures;

    Eigen::Matrix<float, 5, 5> imgPatch;
};

} // namespace slam

#endif