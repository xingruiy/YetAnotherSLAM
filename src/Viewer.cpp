#include "Viewer.h"

#define ZMIN 0.1
#define ZMAX 1000
#define ENTER_KEY 13

Viewer::Viewer(const string &strSettingFile, FullSystem *pSys, Map *pMap)
    : mpSystem(pSys), mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingFile, cv::FileStorage::READ);

    if (fSettings.isOpened())
    {
        mImgWidth = fSettings["Camera.width"];
        mImgHeight = fSettings["Camera.height"];
        mWinWidth = fSettings["Viewer.width"];
        mWinHeight = fSettings["Viewer.height"];

        double fx = fSettings["Camera.fx"];
        double fy = fSettings["Camera.fy"];
        double cx = fSettings["Camera.cx"];
        double cy = fSettings["Camera.cy"];
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0, 0) = fx;
        K(1, 1) = fy;
        K(0, 2) = cx;
        K(1, 2) = cy;
        mKinv = K.inverse();

        mPointSize = fSettings["Viewer.point_size"];
        int nRGB = fSettings["Camera.RGB"];
        mbRGB = (nRGB != 0);
    }

    mTcw = Eigen::Matrix4d::Identity();

    const char vertexShader[] =
        "#version 330\n"
        "\n"
        "layout(location = 0) in vec3 position;\n"
        "layout(location = 1) in vec3 a_normal;\n"
        "uniform mat4 mvpMat;\n"
        "uniform mat4 mMat;\n"
        "out vec3 shaded_colour;\n"
        "\n"
        "void main(void) {\n"
        "    gl_Position = mvpMat * mMat * vec4(position, 1.0);\n"
        "    vec3 lightpos = vec3(5, 5, 5);\n"
        "    const float ka = 0.3;\n"
        "    const float kd = 0.5;\n"
        "    const float ks = 0.2;\n"
        "    const float n = 20.0;\n"
        "    const float ax = 1.0;\n"
        "    const float dx = 1.0;\n"
        "    const float sx = 1.0;\n"
        "    const float lx = 1.0;\n"
        "    vec3 L = normalize(lightpos - position);\n"
        "    vec3 V = normalize(vec3(0.0) - position);\n"
        "    vec3 R = normalize(2 * a_normal * dot(a_normal, L) - L);\n"
        "    float i1 = ax * ka * dx;\n"
        "    float i2 = lx * kd * dx * max(0.0, dot(a_normal, L));\n"
        "    float i3 = lx * ks * sx * pow(max(0.0, dot(R, V)), n);\n"
        "    float Ix = max(0.0, min(255.0, i1 + i2 + i3));\n"
        "    shaded_colour = vec3(Ix, Ix, Ix);\n"
        "}\n";

    const char fragShader[] =
        "#version 330\n"
        "\n"
        "in vec3 shaded_colour;\n"
        "out vec4 colour_out;\n"
        "void main(void) {\n"
        "    colour_out = vec4(shaded_colour, 1);\n"
        "}\n";
}

void Viewer::Spin()
{
    pangolin::CreateWindowAndBind("VIEWER", mWinWidth, mWinHeight);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuPauseTracking("menu.Pause System", true, true);
    pangolin::Var<bool> menuResetSystem("menu.Reset System", true, false);
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuDrawMapPoints("menu.Draw Map Points", true, true);
    pangolin::Var<bool> menuDrawKeyFrame("menu.Draw Key Frames", true, true);
    pangolin::RegisterKeyPressCallback(13, pangolin::ToggleVarFunctor("menu.Pause System"));

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    // Initialize textures
    mImageRGB.Reinitialise(mImgWidth, mImgHeight, GL_RGB, true, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    mImageDepth.Reinitialise(mImgWidth, mImgHeight, GL_RGBA, true, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        if (menuPauseTracking)
            mpSystem->SetToPause();
        else
            mpSystem->SetToUnPause();

        pangolin::glDrawFrustum(mKinv, mImgWidth, mImgHeight, mTcw, 0.1);

        if (menuDrawMapPoints)
            DrawMapPoints();

        if (menuDrawKeyFrame)
            DrawKeyFrames();

        if (pangolin::Pushed(menuResetSystem))
            mpSystem->Reset();

        pangolin::FinishFrame();
    }

    mpSystem->SetToFinish();
}

void Viewer::SetCurrentCameraPose(const Eigen::Matrix4d &Tcw)
{
    std::unique_lock<std::mutex> lock(mCurrentPoseMutex);
    this->mTcw = Tcw;
}

void Viewer::SetCurrentImage(const cv::Mat &imRGB)
{
    mImageRGB.Upload(imRGB.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void Viewer::SetCurrentDepth(const cv::Mat &imDepth)
{
}

void Viewer::DrawMapPoints()
{
    vector<MapPoint *> vpMPs = mpMap->GetAllMapPoints();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        if (!vpMPs[i])
            continue;
        Eigen::Vector3d &pos = vpMPs[i]->mWorldPos;
        glVertex3f(pos(0), pos(1), pos(2));
    }

    glEnd();
}

void Viewer::DrawKeyFrames()
{
    vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        if (!vpKFs[i])
            continue;
        KeyFrame *pKF = vpKFs[i];
        pangolin::glDrawFrustum(mKinv, mImgWidth, mImgHeight, pKF->mTcw.matrix(), 0.1);
    }
}

///////////////////////////////////////////////////

// MapViewer::MapViewer(int w, int h, int fW, int fH, Eigen::Matrix3d &K)
//     : numTriangles(0), maxNumTriangles(20000000), K(K), Kinv(K.inverse()),
//       frameWidth(fW), frameHeight(fH), requestDebugMode(false), requestTestNextKF(false)
// {
//     pangolin::CreateWindowAndBind("MAP VIEWER", w, h);

//     mainCamera = std::make_shared<pangolin::OpenGlRenderState>(
//         pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, ZMIN, ZMAX),
//         pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

//     glEnable(GL_DEPTH_TEST);
//     glEnable(GL_BLEND);

//     setupDisplay();
//     setupKeyBindings();
//     initializeTextures();
//     initializeBuffers();
//     initializePrograms();
// }

// MapViewer::~MapViewer()
// {
//     pangolin::DestroyWindow("MAP VIEWER");
// }

// void MapViewer::setupDisplay()
// {
//     auto MenuDividerLeft = pangolin::Attach::Pix(200);
//     float RightSideBarDividerLeft = 0.75f;

//     modelView = &pangolin::Display("Local Map");
//     modelView->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft).SetHandler(new pangolin::Handler3D(*mainCamera));
//     sidebarView = &pangolin::Display("Right Side Bar");
//     sidebarView->SetBounds(0, 1, RightSideBarDividerLeft, 1);

//     colourView = &pangolin::Display("RGB");
//     colourView->SetBounds(0, 0.5, 0, 0.5);
//     keyPointView = &pangolin::Display("Key Point");
//     keyPointView->SetBounds(0, 0.5, 0.5, 1);

//     subBarView = &pangolin::Display("Sub Side Bar");
//     subBarView->SetBounds(0, 0.33, 0, 1);
//     matchedView = &pangolin::Display("Matched Point");
//     matchedView->SetBounds(0.33, 0.66, 0, 1);
//     depthView = &pangolin::Display("Depth");
//     depthView->SetBounds(0.66, 1, 0, 1);

//     subBarView->AddDisplay(*colourView);
//     subBarView->AddDisplay(*keyPointView);
//     sidebarView->AddDisplay(*depthView);
//     sidebarView->AddDisplay(*subBarView);
//     sidebarView->AddDisplay(*matchedView);

//     pangolin::CreatePanel("Menu").SetBounds(0, 1, 0, MenuDividerLeft);

//     resetBtn = std::make_shared<pangolin::Var<bool>>("Menu.RESET", false, false);
//     saveMapToDiskBtn = std::make_shared<pangolin::Var<bool>>("Menu.Save Map", false, false);
//     readMapFromDiskBtn = std::make_shared<pangolin::Var<bool>>("Menu.Read Map", false, false);
//     pauseSystemBox = std::make_shared<pangolin::Var<bool>>("Menu.PAUSE", true, true);
//     displayColourBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Image", true, true);
//     displayDepthBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Depth", true, true);
//     displayModelBox = std::make_shared<pangolin::Var<bool>>("Menu.View Model", true, true);
//     enableMappingBox = std::make_shared<pangolin::Var<bool>>("Menu.Enable Mapping", true, true);
//     displayFrameHistoryBox = std::make_shared<pangolin::Var<bool>>("Menu.Trajectory", true, true);
//     displayPointBox = std::make_shared<pangolin::Var<bool>>("Menu.Diplay Points", true, true);
//     displayKFHistoryBox = std::make_shared<pangolin::Var<bool>>("Menu.Display Keyframes", true, true);
//     localizationMode = std::make_shared<pangolin::Var<bool>>("Menu.Localization Mode", false, true);
//     allowMatchingAmbiguity = std::make_shared<pangolin::Var<bool>>("Menu.Graph Matching Mode", false, true);
//     incorporateNormal = std::make_shared<pangolin::Var<bool>>("Menu.Incorporate Normal", false, true);
//     displayMatchedPoints = std::make_shared<pangolin::Var<bool>>("Menu.Display Matchings", false, true);
//     enteringDebuggingModeBtn = std::make_shared<pangolin::Var<bool>>("Menu.Debug Mode", false, false);
//     testNextKeyframeBtn = std::make_shared<pangolin::Var<bool>>("Menu.Test Next", false, false);
// }

// void MapViewer::setupKeyBindings()
// {
//     // reset system
//     pangolin::RegisterKeyPressCallback('r', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
//     pangolin::RegisterKeyPressCallback('R', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
//     // pause / unpause the system
//     pangolin::RegisterKeyPressCallback(ENTER_KEY, pangolin::ToggleVarFunctor("Menu.PAUSE"));
//     // toggle localization mode
//     pangolin::RegisterKeyPressCallback('l', pangolin::ToggleVarFunctor("Menu.Localization Mode"));
//     pangolin::RegisterKeyPressCallback('L', pangolin::ToggleVarFunctor("Menu.Localization Mode"));
//     // toggle graph matching mode
//     pangolin::RegisterKeyPressCallback('g', pangolin::ToggleVarFunctor("Menu.Graph Matching Mode"));
//     pangolin::RegisterKeyPressCallback('G', pangolin::ToggleVarFunctor("Menu.Graph Matching Mode"));
//     // toggle normal
//     pangolin::RegisterKeyPressCallback('n', pangolin::ToggleVarFunctor("Menu.Incorporate Normal"));
//     pangolin::RegisterKeyPressCallback('N', pangolin::ToggleVarFunctor("Menu.Incorporate Normal"));
// }

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