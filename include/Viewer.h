#pragma once

#include <pangolin/pangolin.h>
#include <mutex>

#include "FullSystem.h"

class FullSystem;

class Viewer
{
public:
    Viewer(const string &strSettingFile, FullSystem *pSys, Map *pMap);

    void Spin();

    void SetCurrentCameraPose(const Eigen::Matrix4d &Tcw);

    void SetCurrentImage(const cv::Mat &imRGB);

    void SetCurrentDepth(const cv::Mat &imDepth);

private:
    void DrawMapPoints();

    // System components
    FullSystem *mpSystem;
    Map *mpMap;

    // Camera calibration
    Eigen::Matrix3d mKinv;
    Eigen::Matrix4d mTcw;

    std::mutex mCurrentPoseMutex;

    // Image allignment
    bool mbRGB;

    // Image and Window size
    int mImgWidth, mImgHeight;
    int mWinWidth, mWinHeight;

    float mPointSize;

    pangolin::GlTexture mImageRGB;
    pangolin::GlTexture mImageDepth;
    pangolin::GlTexture mImgKeyPoint;
    pangolin::GlTexture mImgKeyPoint2;
};

// class MapViewer
// {
//     pangolin::GlTexture colourImage;
//     pangolin::GlTexture keyPointImage;
//     pangolin::GlTexture matchedImage;
//     pangolin::GlTexture depthImage;
//     pangolin::GlTexture denseMapImage;

//     pangolin::View *sidebarView;
//     pangolin::View *colourView;
//     pangolin::View *keyPointView;
//     pangolin::View *matchedView;
//     pangolin::View *subBarView;
//     pangolin::View *depthView;
//     pangolin::View *modelView;
//     pangolin::View *menuView;
//     std::shared_ptr<pangolin::OpenGlRenderState> mainCamera;

//     std::shared_ptr<pangolin::Var<bool>> resetBtn;
//     std::shared_ptr<pangolin::Var<bool>> saveMapToDiskBtn;
//     std::shared_ptr<pangolin::Var<bool>> readMapFromDiskBtn;
//     std::shared_ptr<pangolin::Var<bool>> pauseSystemBox;
//     std::shared_ptr<pangolin::Var<bool>> displayColourBox;
//     std::shared_ptr<pangolin::Var<bool>> displayDepthBox;
//     std::shared_ptr<pangolin::Var<bool>> displayModelBox;
//     std::shared_ptr<pangolin::Var<bool>> enableMappingBox;
//     std::shared_ptr<pangolin::Var<bool>> displayKFHistoryBox;
//     std::shared_ptr<pangolin::Var<bool>> displayFrameHistoryBox;
//     std::shared_ptr<pangolin::Var<bool>> displayPointBox;
//     std::shared_ptr<pangolin::Var<bool>> localizationMode;
//     std::shared_ptr<pangolin::Var<bool>> incorporateNormal;
//     std::shared_ptr<pangolin::Var<bool>> allowMatchingAmbiguity;
//     std::shared_ptr<pangolin::Var<bool>> displayMatchedPoints;
//     std::shared_ptr<pangolin::Var<bool>> enteringDebuggingModeBtn;
//     std::shared_ptr<pangolin::Var<bool>> testNextKeyframeBtn;

//     GLuint vaoPhong;
//     GLuint vaoColour;
//     pangolin::GlSlProgram phongProgram;
//     pangolin::GlBufferCudaPtr vertexBuffer;
//     pangolin::GlBufferCudaPtr normalBuffer;
//     pangolin::GlBufferCudaPtr colourBuffer;
//     std::shared_ptr<pangolin::CudaScopedMappedPtr> vertexBufferPtr;
//     std::shared_ptr<pangolin::CudaScopedMappedPtr> normalBufferPtr;
//     std::shared_ptr<pangolin::CudaScopedMappedPtr> colourBufferPtr;

//     size_t maxNumTriangles;
//     size_t numTriangles;

//     void setupDisplay();
//     void initializePrograms();
//     void initializeTextures();
//     void initializeBuffers();
//     void setupKeyBindings();
//     void checkButtonsAndBoxes();

//     std::vector<Eigen::Vector3f> activePoints;
//     std::vector<Eigen::Vector3f> stablePoints;
//     std::vector<Eigen::Vector3f> matchedPoints;
//     std::vector<Eigen::Vector3f> matchingLines;
//     std::vector<Eigen::Vector3f> matchedFramePoints;
//     std::vector<Eigen::Vector3f> rawFrameHistory;
//     std::vector<Eigen::Matrix4f> optimizedKeyFramePose;
//     std::vector<Eigen::Matrix4f> rawKeyFrameHistory;
//     std::vector<Eigen::Vector3f> frameHistory;
//     std::vector<Eigen::Matrix4f> keyFrameHistory;
//     std::vector<Eigen::Matrix4f> relocHypotheses;

//     bool requestSystemReset;
//     bool requestDebugMode;
//     bool requestTestNextKF;

//     // draw calls
//     void drawLocalMap();

//     int systemState;
//     Eigen::Matrix3d K, Kinv;
//     int frameWidth;
//     int frameHeight;

//     Sophus::SE3d RTLocalToGlobal;
//     Sophus::SE3d currentCameraPose;

// public:
//     MapViewer(int w, int h, int fW, int fH, Eigen::Matrix3d &K);
//     ~MapViewer();

//     void resetViewer();
//     void renderView();

//     void setFrameHistory(const std::vector<Sophus::SE3d> &history);
//     void setKeyFrameHistory(const std::vector<Sophus::SE3d> &history);
//     void setActivePoints(const std::vector<Eigen::Vector3f> &points);
//     void setStablePoints(const std::vector<Eigen::Vector3f> &points);
//     void setMatchedPoints(const std::vector<Eigen::Vector3f> &points);
//     void setMatchingLines(const std::vector<Eigen::Vector3f> &lines);
//     void setMatchedFramePoints(const std::vector<Eigen::Vector3f> &points);
//     void setRelocalizationHypotheses(std::vector<Sophus::SE3d> &H);

//     bool isResetRequested();
//     bool isDebugRequested();
//     bool isNextKFRequested();
//     bool paused() const;
//     bool isLocalizationMode() const;
//     bool mappingEnabled() const;
//     bool isGraphMatchingMode() const;
//     bool shouldCalculateNormal() const;

//     void setColourImage(cv::Mat image);
//     void setDepthImage(cv::Mat image);
//     void setDenseMapImage(cv::Mat image);
//     void setKeyPointImage(cv::Mat image);
//     void setMatchedPointImage(cv::Mat image);

//     void setMeshSizeToRender(size_t size);
//     void getMeshBuffer(float *&vbuffer, float *&nbuffer, size_t &bufferSize);
//     void setCurrentState(int state);

//     void addTrackingResult(const Sophus::SE3d &T);
//     void addRawKeyFramePose(const Sophus::SE3d &T);
//     void addOptimizedKFPose(const Sophus::SE3d T);
//     void setCurrentCamera(const Sophus::SE3d &T);
//     void setRTLocalToGlobal(const Sophus::SE3d &T);

//     inline bool displayImageRGB() const { return *displayColourBox; }
//     inline bool displayImageDepth() const { return *displayDepthBox; }
// };