#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <mutex>

#include "System.h"
#include "MapDrawer.h"
#include "GlobalDef.h"

namespace SLAM
{

class System;

class Viewer
{
public:
    Viewer(System *pSystem, MapDrawer *pMapDrawer);

    void Run();
    void reset();

    void setLiveImage(const cv::Mat &ImgRGB);
    void setLiveDepth(const cv::Mat &ImgDepth);
    void setLivePose(const Eigen::Matrix4d &Tcw);
    void setReferenceFramePose(const Eigen::Matrix4d &Tcw);

private:
    MapDrawer *mpMapDrawer;
    System *mpSystem;
    Eigen::Matrix3f calibInv;

    void renderImagesToScreen();
    void renderLiveCameraFrustum();
    void renderMapPoints(const int &PointSize, const bool &drawImmature);
    void renderKeyframes();

    bool needUpdateImage;
    bool needUpdateDepth;

    pangolin::GlTexture mTextureColour;
    pangolin::GlTexture mTextureDepth;

    pangolin::View *rightSideBar;
    pangolin::View *imageViewer;
    pangolin::View *mapViewer;
    pangolin::View *depthViewer;

    cv::Mat cvImage8UC3;  // Colour image
    cv::Mat cvImage32FC1; // Depth image

    std::mutex mPoseMutex;
    unsigned long ref_id;
    Eigen::Matrix4f T_frame_ref;
    Eigen::Matrix4f T_ref_world;
    Eigen::Matrix4f T_frame_world;

    int width, height;
    Eigen::Matrix4d mTcw;
    Eigen::Matrix3d mCalib;

    pangolin::GlSlProgram shader;
    pangolin::GlBufferCudaPtr vertexBuffer;
    pangolin::GlBufferCudaPtr normalBuffer;
    pangolin::GlBufferCudaPtr colourBuffer;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> vertexBufferPtr;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> normalBufferPtr;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> colourBufferPtr;
};

} // namespace SLAM

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