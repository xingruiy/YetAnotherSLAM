#include "mapViewer/mapViewer.h"
#include "mapViewer/shader.h"

#define ZMIN 0.1
#define ZMAX 1000
#define ENTER_KEY 13

MapViewer::MapViewer(int w, int h, int fW, int fH, Mat33d &K)
    : numTriangles(0), maxNumTriangles(20000000), K(K), Kinv(K.inverse()),
      frameWidth(fW), frameHeight(fH)
{
    pangolin::CreateWindowAndBind("MAP VIEWER", w, h);

    mainCamera = std::make_shared<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, ZMIN, ZMAX),
        pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);

    setupDisplay();
    setupKeyBindings();
    initializeTextures();
    initializeBuffers();
    initializePrograms();
}

MapViewer::~MapViewer()
{
    pangolin::DestroyWindow("MAP VIEWER");
}

void MapViewer::setupDisplay()
{
    auto MenuDividerLeft = pangolin::Attach::Pix(200);
    float RightSideBarDividerLeft = 0.7f;

    modelView = &pangolin::Display("Local Map");
    modelView->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft).SetHandler(new pangolin::Handler3D(*mainCamera));
    sidebarView = &pangolin::Display("Right Side Bar");
    sidebarView->SetBounds(0, 1, RightSideBarDividerLeft, 1);
    colourView = &pangolin::Display("RGB");
    colourView->SetBounds(0, 0.5, 0, 1);
    depthView = &pangolin::Display("Depth");
    depthView->SetBounds(0.5, 1, 0, 1);

    sidebarView->AddDisplay(*colourView);
    sidebarView->AddDisplay(*depthView);

    pangolin::CreatePanel("Menu").SetBounds(0, 1, 0, MenuDividerLeft);

    resetBtn = std::make_shared<pangolin::Var<bool>>("Menu.RESET", false, false);
    saveMapToDiskBtn = std::make_shared<pangolin::Var<bool>>("Menu.Save Map", false, false);
    readMapFromDiskBtn = std::make_shared<pangolin::Var<bool>>("Menu.Read Map", false, false);
    pauseSystemBox = std::make_shared<pangolin::Var<bool>>("Menu.PAUSE", true, true);
    displayColourBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Image", true, true);
    displayDepthBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Depth", true, true);
    displayModelBox = std::make_shared<pangolin::Var<bool>>("Menu.View Model", true, true);
    enableMappingBox = std::make_shared<pangolin::Var<bool>>("Menu.Current Camera", false, true);
    displayFrameHistoryBox = std::make_shared<pangolin::Var<bool>>("Menu.Trajectory", true, true);
    displayActivePointsBox = std::make_shared<pangolin::Var<bool>>("Menu.Active Points", true, true);
    displayStablePointsBox = std::make_shared<pangolin::Var<bool>>("Menu.Stable Points", true, true);
    displayKFHistoryBox = std::make_shared<pangolin::Var<bool>>("Menu.Key Frame Frustum", true, true);
}

void MapViewer::setupKeyBindings()
{
    // reset system
    pangolin::RegisterKeyPressCallback('r', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    pangolin::RegisterKeyPressCallback('R', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    // pause / unpause the system
    pangolin::RegisterKeyPressCallback(ENTER_KEY, pangolin::ToggleVarFunctor("Menu.PAUSE"));
}

void MapViewer::initializePrograms()
{
    phongProgram.AddShader(
        pangolin::GlSlShaderType::GlSlVertexShader,
        vertexShader);

    phongProgram.AddShader(
        pangolin::GlSlShaderType::GlSlFragmentShader,
        fragShader);

    phongProgram.Link();
}

void MapViewer::initializeTextures()
{
    colourImage.Reinitialise(
        640, 480,
        GL_RGB,
        true,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);

    depthImage.Reinitialise(
        640, 480,
        GL_RGBA,
        true,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        NULL);

    denseMapImage.Reinitialise(
        640, 480,
        GL_RGBA,
        true,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        NULL);
}

void MapViewer::initializeBuffers()
{
    auto size = sizeof(float) * 9 * maxNumTriangles;

    vertexBuffer.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    normalBuffer.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    colourBuffer.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    vertexBufferPtr = std::make_shared<pangolin::CudaScopedMappedPtr>(vertexBuffer);
    normalBufferPtr = std::make_shared<pangolin::CudaScopedMappedPtr>(normalBuffer);
    colourBufferPtr = std::make_shared<pangolin::CudaScopedMappedPtr>(colourBuffer);

    glGenVertexArrays(1, &vaoPhong);
    glBindVertexArray(vaoPhong);

    vertexBuffer.Bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    normalBuffer.Bind();
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    normalBuffer.Unbind();
    glBindVertexArray(0);
}

void MapViewer::resetViewer()
{
}

void MapViewer::setColourImage(Mat image)
{
    colourImage.Upload(image.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void MapViewer::setDepthImage(Mat image)
{
    depthImage.Upload(image.data, GL_RGBA, GL_UNSIGNED_BYTE);
}

void MapViewer::setDenseMapImage(Mat image)
{
    denseMapImage.Upload(image.data, GL_RGBA, GL_UNSIGNED_BYTE);
}

void MapViewer::renderView()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.f, 0.f, 0.f, 1.f);

    checkButtonsAndBoxes();

    if (*displayColourBox && colourView)
    {
        colourView->Activate();
        colourImage.RenderToViewportFlipY();
    }

    if (*displayDepthBox && depthView)
    {
        depthView->Activate();
        depthImage.RenderToViewportFlipY();
    }

    if (*displayModelBox && modelView)
    {
        modelView->Activate(*mainCamera);
        drawLocalMap();
    }

    if (*displayFrameHistoryBox && modelView)
    {
        modelView->Activate(*mainCamera);
        glColor3f(1.f, 0.f, 0.f);
        pangolin::glDrawLineStrip(rawFrameHistory);
        glColor4f(1.f, 1.f, 1.f, 1.f);
    }

    if (*displayActivePointsBox && modelView)
    {
        modelView->Activate(*mainCamera);
        glPointSize(3.f);
        glColor3f(0.f, 1.f, 0.f);
        pangolin::glDrawPoints(activePoints);
        glPointSize(1.f);
        glColor4f(1.f, 1.f, 1.f, 1.f);
    }

    if (*displayStablePointsBox && modelView)
    {
        modelView->Activate(*mainCamera);
        glPointSize(3.f);
        glColor3f(1.f, 0.f, 0.f);
        pangolin::glDrawPoints(stablePoints);
        glPointSize(1.f);
        glColor4f(1.f, 1.f, 1.f, 1.f);
    }

    if (*displayKFHistoryBox && modelView)
    {
        modelView->Activate(*mainCamera);
        glColor3f(1.f, 0.f, 0.f);
        for (auto T : rawKeyFrameHistory)
            pangolin::glDrawFrustum<float>(Kinv.cast<float>(), frameWidth, frameHeight, T, 0.01f);

        glColor3f(0.f, 1.f, 0.f);
        for (auto T : keyFrameHistory)
            pangolin::glDrawFrustum<float>(Kinv.cast<float>(), frameWidth, frameHeight, T, 0.01f);
        glColor4f(1.f, 1.f, 1.f, 1.f);
    }

    pangolin::FinishFrame();
}

void MapViewer::checkButtonsAndBoxes()
{
    if (pangolin::Pushed(*resetBtn))
        requestSystemReset = true;
}

bool MapViewer::isResetRequested()
{
    if (requestSystemReset)
    {
        requestSystemReset = false;
        return true;
    }
    else
        return false;
}

bool MapViewer::paused() const
{
    return *pauseSystemBox;
}

void MapViewer::drawLocalMap()
{
    if (numTriangles == 0)
        return;

    phongProgram.Bind();
    glBindVertexArray(vaoPhong);
    phongProgram.SetUniform("mvpMat", mainCamera->GetProjectionModelViewMatrix());
    glDrawArrays(GL_TRIANGLES, 0, numTriangles * 3);
    glBindVertexArray(0);
    phongProgram.Unbind();
}

void MapViewer::setRawFrameHistory(const std::vector<SE3> &history)
{
    rawFrameHistory.clear();
    for (auto T : history)
        rawFrameHistory.push_back(T.translation().cast<float>());
}

void MapViewer::setRawKeyFrameHistory(const std::vector<SE3> &history)
{
    rawKeyFrameHistory.clear();
    for (auto T : history)
        rawKeyFrameHistory.push_back(T.matrix().cast<float>());
}

void MapViewer::setFrameHistory(const std::vector<SE3> &history)
{
    frameHistory.clear();
    for (auto T : history)
        frameHistory.push_back(T.translation().cast<float>());
}

void MapViewer::setKeyFrameHistory(const std::vector<SE3> &history)
{
    // std::cout << history.size() << std::endl;
    keyFrameHistory.clear();
    for (auto T : history)
        keyFrameHistory.push_back(T.matrix().cast<float>());
}

void MapViewer::getMeshBuffer(float *&vbuffer, float *&nbuffer, size_t &bufferSize)
{
    vbuffer = (float *)**vertexBufferPtr;
    nbuffer = (float *)**normalBufferPtr;
    bufferSize = maxNumTriangles;
}

void MapViewer::setMeshSizeToRender(size_t size)
{
    numTriangles = size;
}

void MapViewer::setActivePoints(const std::vector<Vec3f> &points)
{
    activePoints = points;
}

void MapViewer::setStablePoints(const std::vector<Vec3f> &points)
{
    stablePoints = points;
}
