#include "Viewer.h"

namespace SLAM
{

Viewer::Viewer(System *pSystem, MapDrawer *pMapDrawer)
    : mpSystem(pSystem), mpMapDrawer(pMapDrawer),
      mTcw(Eigen::Matrix4d::Identity()),
      needUpdateImage(false),
      needUpdateDepth(false)
{
    width = g_width[0];
    height = g_height[0];
    mCalib = g_calib[0].cast<double>();
    calibInv = g_calibInv[0];
}

void Viewer::Run()
{
    pangolin::CreateWindowAndBind("SLAM", 1920, 1080);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto RenderState = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(640, 480, g_fx[0], g_fy[0], g_cx[0], g_cy[0], 0.1, 1000),
        pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

    auto MenuDividerLeft = pangolin::Attach::Pix(250);
    float RightSideBarDividerLeft = 0.75f;

    mapViewer = &pangolin::Display("Map");
    mapViewer->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft)
        .SetHandler(new pangolin::Handler3D(RenderState));
    rightSideBar = &pangolin::Display("RightBar");
    rightSideBar->SetBounds(0, 1, RightSideBarDividerLeft, 1);

    imageViewer = &pangolin::Display("RGB");
    imageViewer->SetBounds(0, 0.33, 0, 1);
    depthViewer = &pangolin::Display("Depth");
    depthViewer->SetBounds(0.33, 0.66, 0, 1);

    rightSideBar->AddDisplay(*imageViewer);
    rightSideBar->AddDisplay(*depthViewer);

    // Create textures
    mTextureColour.Reinitialise(width, height, GL_RGB, true, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    mTextureDepth.Reinitialise(width, height, GL_LUMINANCE, true, 0, GL_LUMINANCE, GL_FLOAT, NULL);

    // Create menus
    pangolin::CreatePanel("menu").SetBounds(0, 1, 0, MenuDividerLeft);
    pangolin::Var<bool> varReset = pangolin::Var<bool>("menu.reset", false, false);
    pangolin::Var<bool> varRunning = pangolin::Var<bool>("menu.Running", g_bSystemRunning, true);
    pangolin::RegisterKeyPressCallback(13, pangolin::ToggleVarFunctor("menu.Running"));
    pangolin::Var<bool> varShowKeyFrames = pangolin::Var<bool>("menu.Display KeyFrames", true, true);
    pangolin::Var<bool> varShowKFGraph = pangolin::Var<bool>("menu.Display Covisibility Graph", true, true);
    pangolin::Var<bool> varShowMapPoints = pangolin::Var<bool>("menu.Display MapPoints", true, true);
    pangolin::Var<int> varPointSize = pangolin::Var<int>("menu.Point Size", g_pointSize, 1, 10);
    pangolin::Var<int> varCovMapDensity = pangolin::Var<int>("menu.Covisibility Map Density", 10, 1, 50);

    while (!pangolin::ShouldQuit())
    {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (pangolin::Pushed(varReset))
            mpSystem->reset();

        g_bSystemRunning = varRunning;
        renderImagesToScreen();

        mapViewer->Activate(RenderState);
        renderLiveCameraFrustum();
        mpMapDrawer->DrawKeyFrames(varShowKeyFrames, varShowKFGraph, varCovMapDensity);
        if (varShowMapPoints)
            mpMapDrawer->DrawMapPoints(varPointSize);

        pangolin::FinishFrame();
    }

    mpSystem->kill();
}

void Viewer::renderImagesToScreen()
{
    if (needUpdateImage && !cvImage8UC3.empty())
        mTextureColour.Upload(cvImage8UC3.data, GL_RGB, GL_UNSIGNED_BYTE);
    if (needUpdateDepth && !cvImage32FC1.empty())
        mTextureDepth.Upload(cvImage32FC1.data, GL_LUMINANCE, GL_FLOAT);
    needUpdateImage = needUpdateDepth = false;

    imageViewer->Activate();
    mTextureColour.RenderToViewportFlipY();
    depthViewer->Activate();
    mTextureDepth.RenderToViewportFlipY();
}

void Viewer::renderLiveCameraFrustum()
{
    Eigen::Matrix4f T;
    {
        std::unique_lock<std::mutex> lock(mPoseMutex);
        T = T_frame_world;
    }

    glColor3f(1.0, 0.0, 0.0);
    pangolin::glDrawFrustum(calibInv, width, height, T, 0.1f);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
}

void Viewer::setLivePose(const Eigen::Matrix4d &TFrameRef)
{
    // std::unique_lock<std::mutex> lock(mPoseMutex);
    // T_frame_world = T_ref_world * TFrameRef.cast<float>();
    T_frame_world = TFrameRef.cast<float>();
}

void Viewer::setReferenceFramePose(const Eigen::Matrix4d &TRefWorld)
{
    T_ref_world = TRefWorld.cast<float>();
}

void Viewer::setLiveImage(const cv::Mat &ImgRGB)
{
    cvImage8UC3 = ImgRGB;
    needUpdateImage = true;
}

void Viewer::setLiveDepth(const cv::Mat &ImgDepth)
{
    cvImage32FC1 = ImgDepth;
    needUpdateDepth = true;
}

} // namespace SLAM

// void MapViewer::initializePrograms()
// {
//     phongProgram.AddShader(
//         pangolin::GlSlShaderType::GlSlVertexShader,
//         vertexShader);

//     phongProgram.AddShader(
//         pangolin::GlSlShaderType::GlSlFragmentShader,
//         fragShader);

//     phongProgram.Link();
// }

// void MapViewer::initializeTextures()
// {
//     colourImage.Reinitialise(
//         640, 480,
//         GL_RGB,
//         true,
//         0,
//         GL_RGB,
//         GL_UNSIGNED_BYTE,
//         NULL);

//     keyPointImage.Reinitialise(
//         640, 480,
//         GL_RGB,
//         true,
//         0,
//         GL_RGB,
//         GL_UNSIGNED_BYTE,
//         NULL);

//     matchedImage.Reinitialise(
//         640, 480,
//         GL_RGB,
//         true,
//         0,
//         GL_RGB,
//         GL_UNSIGNED_BYTE,
//         NULL);

//     depthImage.Reinitialise(
//         640, 480,
//         GL_RGBA,
//         true,
//         0,
//         GL_RGBA,
//         GL_UNSIGNED_BYTE,
//         NULL);

//     denseMapImage.Reinitialise(
//         640, 480,
//         GL_RGBA,
//         true,
//         0,
//         GL_RGBA,
//         GL_UNSIGNED_BYTE,
//         NULL);
// }

// void MapViewer::initializeBuffers()
// {
//     auto size = sizeof(float) * 9 * maxNumTriangles;

//     vertexBuffer.Reinitialise(
//         pangolin::GlArrayBuffer,
//         size,
//         cudaGLMapFlagsWriteDiscard,
//         GL_STATIC_DRAW);

//     normalBuffer.Reinitialise(
//         pangolin::GlArrayBuffer,
//         size,
//         cudaGLMapFlagsWriteDiscard,
//         GL_STATIC_DRAW);

//     colourBuffer.Reinitialise(
//         pangolin::GlArrayBuffer,
//         size,
//         cudaGLMapFlagsWriteDiscard,
//         GL_STATIC_DRAW);

//     vertexBufferPtr = std::make_shared<pangolin::CudaScopedMappedPtr>(vertexBuffer);
//     normalBufferPtr = std::make_shared<pangolin::CudaScopedMappedPtr>(normalBuffer);
//     colourBufferPtr = std::make_shared<pangolin::CudaScopedMappedPtr>(colourBuffer);

//     glGenVertexArrays(1, &vaoPhong);
//     glBindVertexArray(vaoPhong);

//     vertexBuffer.Bind();
//     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
//     glEnableVertexAttribArray(0);

//     normalBuffer.Bind();
//     glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
//     glEnableVertexAttribArray(1);

//     normalBuffer.Unbind();
//     glBindVertexArray(0);
// }

// void MapViewer::resetViewer()
// {
//     activePoints.clear();
//     stablePoints.clear();
//     rawFrameHistory.clear();
//     rawKeyFrameHistory.clear();
//     frameHistory.clear();
//     keyFrameHistory.clear();
//     RTLocalToGlobal = Sophus::SE3d();
// }

// void MapViewer::setColourImage(Mat image)
// {
//     colourImage.Upload(image.data, GL_RGB, GL_UNSIGNED_BYTE);
// }

// void MapViewer::setKeyPointImage(Mat image)
// {
//     keyPointImage.Upload(image.data, GL_RGB, GL_UNSIGNED_BYTE);
// }

// void MapViewer::setMatchedPointImage(Mat image)
// {
//     matchedImage.Upload(image.data, GL_RGB, GL_UNSIGNED_BYTE);
// }

// void MapViewer::setDepthImage(Mat image)
// {
//     depthImage.Upload(image.data, GL_RGBA, GL_UNSIGNED_BYTE);
// }

// void MapViewer::setDenseMapImage(Mat image)
// {
//     denseMapImage.Upload(image.data, GL_RGBA, GL_UNSIGNED_BYTE);
// }

// void MapViewer::renderView()
// {
//     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//     glClearColor(0.f, 0.f, 0.f, 1.f);

//     checkButtonsAndBoxes();

//     if (*displayColourBox && colourView)
//     {
//         colourView->Activate();
//         colourImage.RenderToViewportFlipY();

//         keyPointView->Activate();
//         keyPointImage.RenderToViewportFlipY();

//         matchedView->Activate();
//         matchedImage.RenderToViewportFlipY();
//     }

//     if (*displayDepthBox && depthView)
//     {
//         depthView->Activate();
//         depthImage.RenderToViewportFlipY();
//     }

//     if (*displayModelBox && modelView)
//     {
//         modelView->Activate(*mainCamera);
//         drawLocalMap();
//     }

//     if (*displayFrameHistoryBox && modelView)
//     {
//         modelView->Activate(*mainCamera);
//         glColor3f(1.f, 0.f, 0.f);
//         pangolin::glDrawLineStrip(rawFrameHistory);
//         glColor4f(1.f, 1.f, 1.f, 1.f);

//         glColor3f(0.f, 1.f, 0.f);
//         pangolin::glDrawLineStrip(frameHistory);
//         glColor4f(1.f, 1.f, 1.f, 1.f);
//     }

//     if (*displayPointBox && modelView)
//     {
//         modelView->Activate(*mainCamera);
//         // glPointSize(3.f);
//         glColor3f(0.f, 1.f, 0.f);
//         pangolin::glDrawPoints(activePoints);
//         // glPointSize(1.f);
//         glColor4f(1.f, 1.f, 1.f, 1.f);
//     }

//     if (*displayMatchedPoints && modelView)
//     {
//         modelView->Activate(*mainCamera);
//         glColor3f(1.f, 0.f, 0.f);
//         glPointSize(3.f);
//         pangolin::glDrawPoints(matchedPoints);
//         glColor3f(1.f, 0.3f, 1.f);
//         pangolin::glDrawPoints(matchedFramePoints);
//         glPointSize(1.f);
//         glColor3f(0.f, 1.0f, 0.f);
//         pangolin::glDrawLines(matchingLines);
//         glColor4f(1.f, 1.f, 1.f, 1.f);
//     }

//     if (*displayKFHistoryBox && modelView)
//     {
//         // modelView->Activate(*mainCamera);
//         // glColor3f(1.f, 0.f, 0.f);
//         // for (auto T : rawKeyFrameHistory)
//         //     pangolin::glDrawFrustum<float>(Kinv.cast<float>(), frameWidth, frameHeight, T, 0.05f);

//         // glColor3f(0.f, 1.f, 0.f);
//         // for (auto T : optimizedKeyFramePose)
//         //     pangolin::glDrawFrustum<float>(Kinv.cast<float>(), frameWidth, frameHeight, T, 0.05f);

//         glColor3f(0.2f, 0.87f, 0.92f);
//         pangolin::glDrawFrustum<float>(Kinv.cast<float>(), frameWidth, frameHeight, currentCameraPose.matrix().cast<float>(), 0.05f);

//         glColor3f(0.71f, 0.26f, 0.92f);
//         for (auto T : relocHypotheses)
//             pangolin::glDrawFrustum<float>(Kinv.cast<float>(), frameWidth, frameHeight, T, 0.05);
//         glColor4f(1.f, 1.f, 1.f, 1.f);
//     }

//     pangolin::FinishFrame();
// }

// void MapViewer::checkButtonsAndBoxes()
// {
//     if (pangolin::Pushed(*resetBtn))
//         requestSystemReset = true;

//     if (pangolin::Pushed(*enteringDebuggingModeBtn))
//         requestDebugMode = true;

//     if (pangolin::Pushed(*testNextKeyframeBtn))
//         requestTestNextKF = true;

//     if (*localizationMode)
//     {
//         *enableMappingBox = false;
//     }
// }

// bool MapViewer::isResetRequested()
// {
//     if (requestSystemReset)
//     {
//         requestSystemReset = false;
//         return true;
//     }
//     else
//         return false;
// }

// bool MapViewer::isDebugRequested()
// {
//     if (requestDebugMode)
//     {
//         requestDebugMode = false;
//         return true;
//     }
//     else
//         return false;
// }

// bool MapViewer::isNextKFRequested()
// {
//     if (requestTestNextKF)
//     {
//         requestTestNextKF = false;
//         return true;
//     }
//     else
//         return false;
// }

// bool MapViewer::paused() const
// {
//     return *pauseSystemBox;
// }

// bool MapViewer::isLocalizationMode() const
// {
//     return *localizationMode;
// }

// bool MapViewer::isGraphMatchingMode() const
// {
//     return *allowMatchingAmbiguity;
// }

// bool MapViewer::shouldCalculateNormal() const
// {
//     return *incorporateNormal;
// }

// bool MapViewer::mappingEnabled() const
// {
//     return *enableMappingBox;
// }

// void MapViewer::drawLocalMap()
// {
//     if (numTriangles == 0)
//         return;

//     phongProgram.Bind();
//     glBindVertexArray(vaoPhong);
//     phongProgram.SetUniform("mvpMat", mainCamera->GetProjectionModelViewMatrix());
//     phongProgram.SetUniform("mMat", RTLocalToGlobal.matrix());
//     glDrawArrays(GL_TRIANGLES, 0, numTriangles * 3);
//     glBindVertexArray(0);
//     phongProgram.Unbind();
// }

// void MapViewer::setFrameHistory(const std::vector<Sophus::SE3d> &history)
// {
//     frameHistory.clear();
//     for (auto T : history)
//         frameHistory.push_back(T.translation().cast<float>());
// }

// void MapViewer::setKeyFrameHistory(const std::vector<Sophus::SE3d> &history)
// {
//     keyFrameHistory.clear();
//     for (auto T : history)
//         keyFrameHistory.push_back(T.matrix().cast<float>());
// }

// void MapViewer::getMeshBuffer(float *&vbuffer, float *&nbuffer, size_t &bufferSize)
// {
//     vbuffer = (float *)**vertexBufferPtr;
//     nbuffer = (float *)**normalBufferPtr;
//     bufferSize = maxNumTriangles;
// }

// void MapViewer::setMeshSizeToRender(size_t size)
// {
//     numTriangles = size;
// }

// void MapViewer::setActivePoints(const std::vector<Eigen::Vector3f> &points)
// {
//     activePoints = points;
// }

// void MapViewer::setStablePoints(const std::vector<Eigen::Vector3f> &points)
// {
//     stablePoints = points;
// }

// void MapViewer::setMatchedPoints(const std::vector<Eigen::Vector3f> &points)
// {
//     matchedPoints = points;
// }

// void MapViewer::setMatchedFramePoints(const std::vector<Eigen::Vector3f> &points)
// {
//     matchedFramePoints = points;
// }

// void MapViewer::setMatchingLines(const std::vector<Eigen::Vector3f> &lines)
// {
//     matchingLines = lines;
// }

// void MapViewer::setCurrentState(int state)
// {
//     systemState = state;
// }

// void MapViewer::setRelocalizationHypotheses(std::vector<Sophus::SE3d> &H)
// {
//     relocHypotheses.clear();
//     for (auto h : H)
//     {
//         relocHypotheses.push_back((h).matrix().cast<float>());
//     }
// }

// void MapViewer::addTrackingResult(const Sophus::SE3d &T)
// {
//     rawFrameHistory.push_back(T.translation().cast<float>());
// }

// void MapViewer::addRawKeyFramePose(const Sophus::SE3d &T)
// {
//     rawKeyFrameHistory.push_back(T.matrix().cast<float>());
// }

// void MapViewer::addOptimizedKFPose(const Sophus::SE3d T)
// {
//     optimizedKeyFramePose.push_back(T.matrix().cast<float>());
// }

// void MapViewer::setCurrentCamera(const Sophus::SE3d &T)
// {
//     currentCameraPose = RTLocalToGlobal * T;
// }

// void MapViewer::setRTLocalToGlobal(const Sophus::SE3d &T)
// {
//     RTLocalToGlobal = RTLocalToGlobal * T;
// }