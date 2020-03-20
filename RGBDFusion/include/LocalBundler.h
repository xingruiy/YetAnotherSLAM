#ifndef LOCAL_BUNDLER
#define LOCAL_BUNDLER

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#define NUM_KF 7
#define IMW 640
#define IMH 480

enum ResidualState
{
    OOB = -1,
    OK,
    Outlier
};

struct RawResidual
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ResidualState state;

    float hw;
    float r;
    int targetIdx;
    bool active;
    float Jpdd;
    Eigen::Matrix<float, 1, 6> JIdxi;
};

struct PointShell
{
    int x, y;
    int hostIdx;
    float idepth;
    float intensity;
    int numResiduals;
    float Hs;
    float bs;
    RawResidual res[NUM_KF];
};

struct FrameShell
{
    unsigned long KFid;
    int arrayIdx;
    cv::cuda::GpuMat image;
    cv::cuda::GpuMat depth;
    cv::cuda::GpuMat dIdx;
    cv::cuda::GpuMat dIdy;
    cv::cuda::GpuMat OGrid;
    cv::cuda::GpuMat Hcc;
    cv::cuda::GpuMat bc;
    Sophus::SE3d Tcw;
};

class LocalBundler
{
public:
    ~LocalBundler();
    LocalBundler(int w, int h, const Eigen::Matrix3f &K);

    void AddKeyFrame(unsigned long KFid, const cv::Mat depth, const cv::Mat image, const Sophus::SE3d &Tcw);
    void AllocatePoints();
    void BundleAdjust(int maxIter = 10);
    void Reset();

private:
    float LineariseAll();
    void AccumulatePointHessian();
    void AccumulateFrameHessian();
    void AccumulateShcurrHessian();
    void AccumulateShcurrResidual();

    void SolveSystem();
    void Marginalization();
    void PopulateOccupancyGrid(FrameShell *F);

    int width;
    int height;
    Eigen::Matrix3f K;
    int frameCount;

    PointShell *points_dev;
    int *N;
    int nPoints;
    int *stack;
    int lastKeyframeIdx;
    Eigen::Vector3f *frameData_dev;
    Sophus::SE3f *constPoseMat;

    std::array<FrameShell *, NUM_KF> frame;
    std::array<Sophus::SE3d, NUM_KF> framePose;
    std::array<cv::cuda::GpuMat, NUM_KF> frameHessian;
    std::array<cv::cuda::GpuMat, NUM_KF> ReduceResidual;
    std::array<cv::cuda::GpuMat, NUM_KF * NUM_KF> reduceHessian;
    std::array<Eigen::Matrix<float, 6, 6>, NUM_KF> frameHesOut;
    std::array<Eigen::Matrix<float, 6, 1>, NUM_KF> frameResOut;
    Eigen::Matrix<float, 6 * NUM_KF, 6 * NUM_KF> reduceHessianOut;
    Eigen::Vector<float, 6 * NUM_KF> ReduceResidualOut;

    cv::cuda::GpuMat FrameHessianR;
    cv::cuda::GpuMat FrameHessianRFinal;
    cv::cuda::GpuMat Residual;
    cv::cuda::GpuMat ResidualFinal;
    cv::cuda::GpuMat EnergySum;
    cv::cuda::GpuMat EnergySumFinal;

    void UpdatePoseMatrix(std::array<Sophus::SE3d, NUM_KF> &poses);
};

#endif