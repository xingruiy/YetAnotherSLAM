#pragma once
#include "utils/numType.h"
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <memory>

class MapViewer
{
    pangolin::GlTexture colourImage;
    pangolin::GlTexture depthImage;
    pangolin::GlTexture denseMapImage;

    pangolin::View *sidebarView;
    pangolin::View *colourView;
    pangolin::View *depthView;
    pangolin::View *localMapView;
    pangolin::View *modelView;
    pangolin::View *menuView;
    std::shared_ptr<pangolin::OpenGlRenderState> mainCamera;

    std::shared_ptr<pangolin::Var<bool>> resetBtn;
    std::shared_ptr<pangolin::Var<bool>> saveMapToDiskBtn;
    std::shared_ptr<pangolin::Var<bool>> readMapFromDiskBtn;
    std::shared_ptr<pangolin::Var<bool>> pauseSystemBox;
    std::shared_ptr<pangolin::Var<bool>> displayColourBox;
    std::shared_ptr<pangolin::Var<bool>> displayDepthBox;
    std::shared_ptr<pangolin::Var<bool>> displayModelBox;
    std::shared_ptr<pangolin::Var<bool>> enableMappingBox;
    std::shared_ptr<pangolin::Var<bool>> displayKFHistoryBox;
    std::shared_ptr<pangolin::Var<bool>> displayFrameHistoryBox;
    std::shared_ptr<pangolin::Var<bool>> displayPointBox;
    std::shared_ptr<pangolin::Var<bool>> localizationMode;
    std::shared_ptr<pangolin::Var<bool>> allowMatchingAmbiguity;

    GLuint vaoPhong;
    GLuint vaoColour;
    pangolin::GlSlProgram phongProgram;
    pangolin::GlBufferCudaPtr vertexBuffer;
    pangolin::GlBufferCudaPtr normalBuffer;
    pangolin::GlBufferCudaPtr colourBuffer;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> vertexBufferPtr;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> normalBufferPtr;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> colourBufferPtr;

    size_t maxNumTriangles;
    size_t numTriangles;

    void setupDisplay();
    void initializePrograms();
    void initializeTextures();
    void initializeBuffers();
    void setupKeyBindings();
    void checkButtonsAndBoxes();

    std::vector<Vec3f> activePoints;
    std::vector<Vec3f> stablePoints;
    std::vector<Vec3f> matchedPoints;
    std::vector<Vec3f> rawFrameHistory;
    std::vector<Mat44f> rawKeyFrameHistory;
    std::vector<Vec3f> frameHistory;
    std::vector<Mat44f> keyFrameHistory;
    std::vector<Mat44f> relocHypotheses;

    bool requestSystemReset;

    // draw calls
    void drawLocalMap();

    int systemState;
    Mat33d K, Kinv;
    int frameWidth;
    int frameHeight;

public:
    MapViewer(int w, int h, int fW, int fH, Mat33d &K);
    ~MapViewer();

    void resetViewer();
    void renderView();

    void setFrameHistory(const std::vector<SE3> &history);
    void setKeyFrameHistory(const std::vector<SE3> &history);
    void setActivePoints(const std::vector<Vec3f> &points);
    void setStablePoints(const std::vector<Vec3f> &points);
    void setMatchedPoints(const std::vector<Vec3f> &points);
    void setRelocalizationHypotheses(std::vector<SE3> &H);

    bool isResetRequested();
    bool paused() const;
    bool isLocalizationMode() const;
    bool mappingEnabled() const;
    bool isGraphMatchingMode() const;

    void setColourImage(Mat image);
    void setDepthImage(Mat image);
    void setDenseMapImage(Mat image);

    void setMeshSizeToRender(size_t size);
    void getMeshBuffer(float *&vbuffer, float *&nbuffer, size_t &bufferSize);
    void setCurrentState(int state);

    void addTrackingResult(const SE3 &T);
    void addRawKeyFramePose(const SE3 &T);
};