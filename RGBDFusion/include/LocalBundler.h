#ifndef LOCAL_BUNDLER
#define LOCAL_BUNDLER

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#define NUM_KF 7

enum PointState
{
    OOB = -1,
    Unchecked,
    OK,
    Discard,
    Marginalized
};

struct RawResidual
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    float hw;
    Eigen::Matrix<float, 2, 1> uv;
    Eigen::Matrix<float, 2, 1> Jpdd;
    Eigen::Matrix<float, 1, 2> JIdp;
    Eigen::Matrix<float, 2, 6> Jpdxi;
};

struct PointShell
{
    PointState state;
    int x, y;
    int frameIdx;
    float idepth;
    float intensity;
    int numResiduals;
    RawResidual res[NUM_KF];
};

struct FrameShell
{
    int idx_in_array;
    unsigned long int id;
    cv::cuda::GpuMat image;
    cv::cuda::GpuMat depth;
    cv::cuda::GpuMat dIdx;
    cv::cuda::GpuMat dIdy;
    cv::cuda::GpuMat OGrid;
    Sophus::SE3d Tcw;
};

class LocalBundler
{
public:
    ~LocalBundler();
    LocalBundler(int w, int h, const Eigen::Matrix3f &K);

    void AddKeyFrame(const cv::Mat depth,
                     const cv::Mat image,
                     const Sophus::SE3d &Tcw);
    void BundleAdjust(const int maxIter = 10);
    void Reset();

private:
    void LineariseAll();
    void Marginalization();
    void CreateNewPoints();
    void CheckProjections(FrameShell &F);
    void PopulateOccupancyGrid(FrameShell &F);

    int width;
    int height;
    Eigen::Matrix3f calib;

    unsigned long int frameCount;

    std::vector<FrameShell> frames;

    int *stack_dev;
    int *stackPtr_dev;
    PointShell *points_dev;
    Sophus::SE3f *posesMatrix_dev;
};

#endif