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
    std::shared_ptr<pangolin::Var<bool>> displayLocalMapBox;
    std::shared_ptr<pangolin::Var<bool>> displayModelBox;
    std::shared_ptr<pangolin::Var<bool>> enableMappingBox;
    std::shared_ptr<pangolin::Var<bool>> displayFrameHistoryBox;

    GLuint vaoPhong;
    GLuint vaoColour;
    pangolin::GlBufferCudaPtr vertexBuffer;
    pangolin::GlBufferCudaPtr normalBuffer;
    pangolin::GlBufferCudaPtr colourBuffer;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> vertexBufferPtr;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> normalBufferPtr;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> colourBufferPtr;

    size_t numTriangles;
    size_t numKeyPoints;
    size_t maxNumTriangles;

    pangolin::GlSlProgram phongProgram;

    void setupDisplay();
    void initializePrograms();
    void initializeTextures();
    void initializeBuffers();
    void setupKeyBindings();
    void checkButtonsAndBoxes();

    std::vector<Vec3f> rawFrameHistory;
    std::vector<Mat44f> rawKeyFrameHistory;
    std::vector<Vec3f> frameHistory;
    std::vector<Mat44f> keyFrameHistory;

    bool requestSystemReset;

    void drawLocalMap();
    void drawFrameHistory();
    float *getVertexBufferPtr();
    float *getNormalBufferPtr();
    uchar *getColourBufferPtr();

public:
    MapViewer(int w, int h);
    void resetViewer();
    void renderView();

    void setRawFrameHistory(const std::vector<SE3> &history);
    void setRawKeyFrameHistory(const std::vector<SE3> &history);
    void setFrameHistory(const std::vector<SE3> &history);
    void setKeyFrameHistory(const std::vector<SE3> &history);

    bool isResetRequested();
    bool isSystemPaused() const;

    void setColourImage(Mat image);
    void setDepthImage(Mat image);
    void setDenseMapImage(Mat image);
};