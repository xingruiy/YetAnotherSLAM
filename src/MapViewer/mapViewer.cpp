#include "MapViewer/mapViewer.h"
#include "MapViewer/shader.h"

#define ZMIN 0.1
#define ZMAX 1000
#define ENTER_KEY 13

MapViewer::MapViewer(int w, int h, int fW, int fH, Mat33d &K)
    : numTriangles(0), maxNumTriangles(20000000), K(K), Kinv(K.inverse()),
      frameWidth(fW), frameHeight(fH), requestDebugMode(false), requestTestNextKF(false)
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
    float RightSideBarDividerLeft = 0.75f;

    modelView = &pangolin::Display("Local Map");
    modelView->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft).SetHandler(new pangolin::Handler3D(*mainCamera));
    sidebarView = &pangolin::Display("Right Side Bar");
    sidebarView->SetBounds(0, 1, RightSideBarDividerLeft, 1);

    colourView = &pangolin::Display("RGB");
    colourView->SetBounds(0, 0.5, 0, 0.5);
    keyPointView = &pangolin::Display("Key Point");
    keyPointView->SetBounds(0, 0.5, 0.5, 1);

    subBarView = &pangolin::Display("Sub Side Bar");
    subBarView->SetBounds(0, 0.33, 0, 1);
    matchedView = &pangolin::Display("Matched Point");
    matchedView->SetBounds(0.33, 0.66, 0, 1);
    depthView = &pangolin::Display("Depth");
    depthView->SetBounds(0.66, 1, 0, 1);

    subBarView->AddDisplay(*colourView);
    subBarView->AddDisplay(*keyPointView);
    sidebarView->AddDisplay(*depthView);
    sidebarView->AddDisplay(*subBarView);
    sidebarView->AddDisplay(*matchedView);

    pangolin::CreatePanel("Menu").SetBounds(0, 1, 0, MenuDividerLeft);

    resetBtn = std::make_shared<pangolin::Var<bool>>("Menu.RESET", false, false);
    saveMapToDiskBtn = std::make_shared<pangolin::Var<bool>>("Menu.Save Map", false, false);
    readMapFromDiskBtn = std::make_shared<pangolin::Var<bool>>("Menu.Read Map", false, false);
    pauseSystemBox = std::make_shared<pangolin::Var<bool>>("Menu.PAUSE", true, true);
    displayColourBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Image", true, true);
    displayDepthBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Depth", true, true);
    displayModelBox = std::make_shared<pangolin::Var<bool>>("Menu.View Model", true, true);
    enableMappingBox = std::make_shared<pangolin::Var<bool>>("Menu.Enable Mapping", true, true);
    displayFrameHistoryBox = std::make_shared<pangolin::Var<bool>>("Menu.Trajectory", true, true);
    displayPointBox = std::make_shared<pangolin::Var<bool>>("Menu.Diplay Points", true, true);
    displayKFHistoryBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Keyframes", true, true);
    localizationMode = std::make_shared<pangolin::Var<bool>>("Menu.Localization Mode", false, true);
    allowMatchingAmbiguity = std::make_shared<pangolin::Var<bool>>("Menu.Graph Matching Mode", false, true);
    incorporateNormal = std::make_shared<pangolin::Var<bool>>("Menu.Incorporate Normal", false, true);
    displayMatchedPoints = std::make_shared<pangolin::Var<bool>>("Menu.Display Matchings", false, true);
    enteringDebuggingModeBtn = std::make_shared<pangolin::Var<bool>>("Menu.Debug Mode", false, false);
    testNextKeyframeBtn = std::make_shared<pangolin::Var<bool>>("Menu.Test Next", false, false);
}

void MapViewer::setupKeyBindings()
{
    // reset system
    pangolin::RegisterKeyPressCallback('r', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    pangolin::RegisterKeyPressCallback('R', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    // pause / unpause the system
    pangolin::RegisterKeyPressCallback(ENTER_KEY, pangolin::ToggleVarFunctor("Menu.PAUSE"));
    // toggle localization mode
    pangolin::RegisterKeyPressCallback('l', pangolin::ToggleVarFunctor("Menu.Localization Mode"));
    pangolin::RegisterKeyPressCallback('L', pangolin::ToggleVarFunctor("Menu.Localization Mode"));
    // toggle graph matching mode
    pangolin::RegisterKeyPressCallback('g', pangolin::ToggleVarFunctor("Menu.Graph Matching Mode"));
    pangolin::RegisterKeyPressCallback('G', pangolin::ToggleVarFunctor("Menu.Graph Matching Mode"));
    // toggle normal
    pangolin::RegisterKeyPressCallback('n', pangolin::ToggleVarFunctor("Menu.Incorporate Normal"));
    pangolin::RegisterKeyPressCallback('N', pangolin::ToggleVarFunctor("Menu.Incorporate Normal"));
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

    keyPointImage.Reinitialise(
        640, 480,
        GL_RGB,
        true,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);

    matchedImage.Reinitialise(
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
    activePoints.clear();
    stablePoints.clear();
    rawFrameHistory.clear();
    rawKeyFrameHistory.clear();
    frameHistory.clear();
    keyFrameHistory.clear();
    RTLocalToGlobal = SE3();
}

void MapViewer::setColourImage(Mat image)
{
    colourImage.Upload(image.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void MapViewer::setKeyPointImage(Mat image)
{
    keyPointImage.Upload(image.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void MapViewer::setMatchedPointImage(Mat image)
{
    matchedImage.Upload(image.data, GL_RGB, GL_UNSIGNED_BYTE);
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

        keyPointView->Activate();
        keyPointImage.RenderToViewportFlipY();

        matchedView->Activate();
        matchedImage.RenderToViewportFlipY();
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

        glColor3f(0.f, 1.f, 0.f);
        pangolin::glDrawLineStrip(frameHistory);
        glColor4f(1.f, 1.f, 1.f, 1.f);
    }

    if (*displayPointBox && modelView)
    {
        modelView->Activate(*mainCamera);
        // glPointSize(3.f);
        glColor3f(0.f, 1.f, 0.f);
        pangolin::glDrawPoints(activePoints);
        // glPointSize(1.f);
        glColor4f(1.f, 1.f, 1.f, 1.f);
    }

    if (*displayMatchedPoints && modelView)
    {
        modelView->Activate(*mainCamera);
        glColor3f(1.f, 0.f, 0.f);
        glPointSize(3.f);
        pangolin::glDrawPoints(matchedPoints);
        glColor3f(1.f, 0.3f, 1.f);
        pangolin::glDrawPoints(matchedFramePoints);
        glPointSize(1.f);
        glColor3f(0.f, 1.0f, 0.f);
        pangolin::glDrawLines(matchingLines);
        glColor4f(1.f, 1.f, 1.f, 1.f);
    }

    if (*displayKFHistoryBox && modelView)
    {
        // modelView->Activate(*mainCamera);
        // glColor3f(1.f, 0.f, 0.f);
        // for (auto T : rawKeyFrameHistory)
        //     pangolin::glDrawFrustum<float>(Kinv.cast<float>(), frameWidth, frameHeight, T, 0.05f);

        // glColor3f(0.f, 1.f, 0.f);
        // for (auto T : optimizedKeyFramePose)
        //     pangolin::glDrawFrustum<float>(Kinv.cast<float>(), frameWidth, frameHeight, T, 0.05f);

        glColor3f(0.2f, 0.87f, 0.92f);
        pangolin::glDrawFrustum<float>(Kinv.cast<float>(), frameWidth, frameHeight, currentCameraPose.matrix().cast<float>(), 0.05f);

        glColor3f(0.71f, 0.26f, 0.92f);
        for (auto T : relocHypotheses)
            pangolin::glDrawFrustum<float>(Kinv.cast<float>(), frameWidth, frameHeight, T, 0.05);
        glColor4f(1.f, 1.f, 1.f, 1.f);
    }

    pangolin::FinishFrame();
}

void MapViewer::checkButtonsAndBoxes()
{
    if (pangolin::Pushed(*resetBtn))
        requestSystemReset = true;

    if (pangolin::Pushed(*enteringDebuggingModeBtn))
        requestDebugMode = true;

    if (pangolin::Pushed(*testNextKeyframeBtn))
        requestTestNextKF = true;

    if (*localizationMode)
    {
        *enableMappingBox = false;
    }
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

bool MapViewer::isDebugRequested()
{
    if (requestDebugMode)
    {
        requestDebugMode = false;
        return true;
    }
    else
        return false;
}

bool MapViewer::isNextKFRequested()
{
    if (requestTestNextKF)
    {
        requestTestNextKF = false;
        return true;
    }
    else
        return false;
}

bool MapViewer::paused() const
{
    return *pauseSystemBox;
}

bool MapViewer::isLocalizationMode() const
{
    return *localizationMode;
}

bool MapViewer::isGraphMatchingMode() const
{
    return *allowMatchingAmbiguity;
}

bool MapViewer::shouldCalculateNormal() const
{
    return *incorporateNormal;
}

bool MapViewer::mappingEnabled() const
{
    return *enableMappingBox;
}

void MapViewer::drawLocalMap()
{
    if (numTriangles == 0)
        return;

    phongProgram.Bind();
    glBindVertexArray(vaoPhong);
    phongProgram.SetUniform("mvpMat", mainCamera->GetProjectionModelViewMatrix());
    phongProgram.SetUniform("mMat", RTLocalToGlobal.matrix());
    glDrawArrays(GL_TRIANGLES, 0, numTriangles * 3);
    glBindVertexArray(0);
    phongProgram.Unbind();
}

void MapViewer::setFrameHistory(const std::vector<SE3> &history)
{
    frameHistory.clear();
    for (auto T : history)
        frameHistory.push_back(T.translation().cast<float>());
}

void MapViewer::setKeyFrameHistory(const std::vector<SE3> &history)
{
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

void MapViewer::setMatchedPoints(const std::vector<Vec3f> &points)
{
    matchedPoints = points;
}

void MapViewer::setMatchedFramePoints(const std::vector<Vec3f> &points)
{
    matchedFramePoints = points;
}

void MapViewer::setMatchingLines(const std::vector<Vec3f> &lines)
{
    matchingLines = lines;
}

void MapViewer::setCurrentState(int state)
{
    systemState = state;
}

void MapViewer::setRelocalizationHypotheses(std::vector<SE3> &H)
{
    relocHypotheses.clear();
    for (auto h : H)
    {
        relocHypotheses.push_back((h).matrix().cast<float>());
    }
}

void MapViewer::addTrackingResult(const SE3 &T)
{
    rawFrameHistory.push_back(T.translation().cast<float>());
}

void MapViewer::addRawKeyFramePose(const SE3 &T)
{
    rawKeyFrameHistory.push_back(T.matrix().cast<float>());
}

void MapViewer::addOptimizedKFPose(const SE3 T)
{
    optimizedKeyFramePose.push_back(T.matrix().cast<float>());
}

void MapViewer::setCurrentCamera(const SE3 &T)
{
    currentCameraPose = RTLocalToGlobal * T;
}

void MapViewer::setRTLocalToGlobal(const SE3 &T)
{
    RTLocalToGlobal = RTLocalToGlobal * T;
}