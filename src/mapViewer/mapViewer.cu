#include "mapViewer/mapViewer.h"
#include "mapViewer/shader.h"

#define ZMIN 0.1
#define ZMAX 1000
#define ENTER_KEY 13

MapViewer::MapViewer(int w, int h)
{
    pangolin::CreateWindowAndBind("MAP VIEWER", w, h);

    mainCamera = std::make_shared<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, ZMIN, ZMAX),
        pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

    glEnable(GL_DEPTH_TEST);

    setupDisplay();
    setupKeyBindings();
    initializeTextures();
    initializeBuffers();
}

void MapViewer::setupDisplay()
{
    auto MenuDividerLeft = pangolin::Attach::Pix(200);
    float RightSideBarDividerLeft = 0.7f;
    pangolin::CreatePanel("Menu").SetBounds(0, 1, 0, MenuDividerLeft);
    resetBtn = std::make_shared<pangolin::Var<bool>>("Menu.RESET", false, false);
    saveMapToDiskBtn = std::make_shared<pangolin::Var<bool>>("Menu.Save Map", false, false);
    readMapFromDiskBtn = std::make_shared<pangolin::Var<bool>>("Menu.Read Map", false, false);
    pauseSystemBox = std::make_shared<pangolin::Var<bool>>("Menu.PAUSE", true, true);
    displayColourBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Image", true, true);
    displayDepthBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Depth", true, true);
    displayLocalMapBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Scene", true, true);
    displayModelBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Mesh", true, true);
    enableMappingBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Camera", false, true);

    sidebarView = &pangolin::Display("Right Side Bar");
    sidebarView->SetBounds(0, 1, RightSideBarDividerLeft, 1);
    colourView = &pangolin::Display("RGB");
    colourView->SetBounds(0, 0.5, 0, 1);
    depthView = &pangolin::Display("Depth");
    depthView->SetBounds(0.5, 1, 0, 1);
    localMapView = &pangolin::Display("Scene");
    localMapView->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft);
    modelView = &pangolin::Display("Mesh");
    modelView->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft).SetHandler(new pangolin::Handler3D(*mainCamera));

    sidebarView->AddDisplay(*colourView);
    sidebarView->AddDisplay(*depthView);
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
    phongProgram.AddShaderFromFile(
        pangolin::GlSlShaderType::GlSlVertexShader,
        vsPhong);

    phongProgram.AddShaderFromFile(
        pangolin::GlSlShaderType::GlSlFragmentShader,
        vsPhong);

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

void MapViewer::setRawFrameHistory(const std::vector<SE3> &history)
{
}

void MapViewer::setRawKeyFrameHistory(const std::vector<SE3> &history)
{
}

void MapViewer::setFrameHistory(const std::vector<SE3> &history)
{
}

void MapViewer::setKeyFrameHistory(const std::vector<SE3> &history)
{
}

void MapViewer::renderView()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.f, 0.f, 0.f, 1.f);

    if (*displayColourBox)
    {
        colourView->Activate();
        colourImage.RenderToViewportFlipY();
    }

    if (*displayDepthBox)
    {
        depthView->Activate();
        depthImage.RenderToViewportFlipY();
    }

    if (*displayLocalMapBox)
    {
        modelView->Activate();
        drawLocalMap();
    }
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

bool MapViewer::isSystemPaused() const
{
    return *pauseSystemBox;
}

void MapViewer::drawLocalMap()
{
    if (numTriangles == 0)
        return;

    phongProgram.Bind();
    glBindVertexArray(vaoPhong);
    phongProgram.SetUniform("mvp_matrix", mainCamera->GetProjectionModelViewMatrix());
    glDrawArrays(GL_TRIANGLES, 0, numTriangles * 9);
    glBindVertexArray(0);
    phongProgram.Unbind();
}

float *MapViewer::getVertexBufferPtr()
{
    return (float *)**vertexBufferPtr;
}

float *MapViewer::getNormalBufferPtr()
{
    return (float *)**normalBufferPtr;
}

uchar *MapViewer::getColourBufferPtr()
{
    return (uchar *)**colourBufferPtr;
}
