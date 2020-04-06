#include "Viewer.h"

namespace slam
{

Viewer::Viewer(System *pSystem, MapDrawer *pMapDrawer)
    : mpSystem(pSystem), mpMapDrawer(pMapDrawer),
      mbNewImage(false), mbNewDepth(false), mbNewKF(false)
{
    width = g_width[0];
    height = g_height[0];
    mCalibInv = g_calibInv[0];
}

void Viewer::Run()
{
    pangolin::CreateWindowAndBind("slam", 1920, 1080);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto RenderState = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(640, 480, g_fx[0], g_fy[0], g_cx[0], g_cy[0], 0.1, 1000),
        pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

    auto MenuDividerLeft = pangolin::Attach::Pix(300);
    float RightSideBarDividerLeft = 0.75f;

    mpMapView = &pangolin::Display("Map");
    mpMapView->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft)
        .SetHandler(new pangolin::Handler3D(RenderState));
    mpRightImageBar = &pangolin::Display("RightBar");
    mpRightImageBar->SetBounds(0, 1, RightSideBarDividerLeft, 1);

    mpCurrentImageView = &pangolin::Display("RGB");
    mpCurrentImageView->SetBounds(0, 0.33, 0, 1);
    mpCurrentDepthView = &pangolin::Display("Depth");
    mpCurrentDepthView->SetBounds(0.33, 0.66, 0, 1);
    mpCurrentKFView = &pangolin::Display("KF");
    mpCurrentKFView->SetBounds(0.66, 1.0, 0, 1);

    mpRightImageBar->AddDisplay(*mpCurrentImageView);
    mpRightImageBar->AddDisplay(*mpCurrentDepthView);
    mpRightImageBar->AddDisplay(*mpCurrentKFView);

    // Create textures
    mTextureKF.Reinitialise(width, height, GL_RGB, true, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    mTextureColour.Reinitialise(width, height, GL_RGB, true, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    mTextureDepth.Reinitialise(width, height, GL_LUMINANCE, true, 0, GL_LUMINANCE, GL_FLOAT, nullptr);

    // Create menus
    pangolin::CreatePanel("menu").SetBounds(0, 1, 0, MenuDividerLeft);
    pangolin::Var<bool> varReset("menu.reset", false, false);
    pangolin::Var<bool> varFuseMap("menu.Fuse Map", false, false);
    pangolin::Var<bool> varRunning("menu.Running", g_bSystemRunning, true);
    pangolin::Var<bool> varShowKeyFrames("menu.Display KeyFrames", true, true);
    pangolin::Var<bool> varShowKFGraph("menu.Display Covisibility Graph", true, true);
    pangolin::Var<bool> varShowMapPoints("menu.Display MapPoints", true, true);
    pangolin::Var<bool> varShowMapStructs("menu.Display MapStructs", false, true);
    pangolin::Var<int> varPointSize("menu.Point Size", g_pointSize, 1, 10);
    pangolin::Var<int> varCovMapDensity("menu.Covisibility Map Density", 10, 1, 50);
    pangolin::Var<int> varDisplayMeshNum("menu.Display Mesh Number", -1, -1, 80);
    pangolin::Var<bool> varSaveMap("menu.Save Map", false, false);
    pangolin::Var<bool> varReadMap("menu.Read Map", false, false);
    pangolin::Var<std::string> strFileName("menu.File Name:", "map.bin");

    pangolin::RegisterKeyPressCallback(13, pangolin::ToggleVarFunctor("menu.Running"));

    mpMapDrawer->LinkGlSlProgram();

    while (!pangolin::ShouldQuit())
    {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (pangolin::Pushed(varFuseMap))
            mpSystem->FuseAllMapStruct();

        if (pangolin::Pushed(varReset))
            mpSystem->reset();

        if (pangolin::Pushed(varSaveMap))
            mpSystem->WriteToFile(strFileName);

        g_bSystemRunning = varRunning;
        RenderImagesToScreen();

        mpMapView->Activate(RenderState);
        RenderLiveCameraFrustum();

        mpMapDrawer->DrawKeyFrames(varShowKeyFrames, varShowKFGraph, varCovMapDensity);

        if (varShowMapPoints)
            mpMapDrawer->DrawMapPoints(varPointSize);

        if (varShowMapStructs)
        {
            pangolin::OpenGlMatrix modelViewMatrix = RenderState.GetProjectionModelViewMatrix();
            mpMapDrawer->DrawMesh(varDisplayMeshNum, modelViewMatrix);
        }

        pangolin::FinishFrame();
    }

    mpSystem->Shutdown();
}

void Viewer::RenderImagesToScreen()
{
    cv::Mat im, imD, imKF;
    bool newImage = false;
    bool newDepth = false;
    bool newKF = false;
    std::vector<bool> vMatches;
    std::vector<cv::KeyPoint> vKeys;

    {
        std::unique_lock<std::mutex> lock(mImageMutex);

        if (mbNewImage)
        {
            mCvImageRGB.copyTo(im);
            newImage = true;
            mbNewImage = false;
        }

        if (mbNewDepth)
        {
            mCvImageDepth.copyTo(imD);
            newDepth = true;
            mbNewDepth = false;
        }

        if (mbNewKF)
        {
            mCvImageKF.copyTo(imKF);
            vKeys = mvCurrentKeys;
            newKF = true;
            mbNewKF = false;
        }
    }

    if (newImage && !im.empty())
    {
        mTextureColour.Upload(im.data,
                              GL_RGB,
                              GL_UNSIGNED_BYTE);
    }

    if (newDepth && !imD.empty())
    {
        mTextureDepth.Upload(imD.data,
                             GL_LUMINANCE,
                             GL_FLOAT);
    }

    if (newKF && !imKF.empty())
    {
        if (imKF.channels() < 3)
            cv::cvtColor(imKF, imKF, CV_GRAY2RGB);

        const float r = 5;
        const int n = vKeys.size();
        for (int i = 0; i < n; i++)
        {
            cv::Point2f pt1, pt2;
            pt1.x = vKeys[i].pt.x - r;
            pt1.y = vKeys[i].pt.y - r;
            pt2.x = vKeys[i].pt.x + r;
            pt2.y = vKeys[i].pt.y + r;

            cv::rectangle(imKF, pt1, pt2, cv::Scalar(0, 255, 0));
            cv::circle(imKF, vKeys[i].pt, 2, cv::Scalar(0, 255, 0), -1);
        }

        mTextureKF.Upload(imKF.data,
                          GL_RGB,
                          GL_UNSIGNED_BYTE);
    }

    mpCurrentImageView->Activate();
    mTextureColour.RenderToViewportFlipY();
    mpCurrentDepthView->Activate();
    mTextureDepth.RenderToViewportFlipY();
    mpCurrentKFView->Activate();
    mTextureKF.RenderToViewportFlipY();
}

void Viewer::RenderLiveCameraFrustum()
{
    Eigen::Matrix4f T;

    {
        std::unique_lock<std::mutex> lock(mmMutexPose);
        T = mCurrentCameraPose;
    }

    glColor3f(1.0, 0.0, 0.0);
    pangolin::glDrawFrustum(mCalibInv, width, height, T, 0.05f);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
}

void Viewer::setLivePose(const Eigen::Matrix4d &T)
{
    mCurrentCameraPose = T.cast<float>();
}

void Viewer::setLiveImage(const cv::Mat &ImgRGB)
{
    std::unique_lock<std::mutex> lock(mImageMutex);
    mCvImageRGB = ImgRGB;
    mbNewImage = true;
}

void Viewer::setKeyFrameImage(const cv::Mat &im, std::vector<cv::KeyPoint> vKeys)
{
    std::unique_lock<std::mutex> lock(mImageMutex);
    mCvImageKF = im;
    mvCurrentKeys = vKeys;
    mbNewKF = true;
}

void Viewer::setLiveDepth(const cv::Mat &ImgDepth)
{
    std::unique_lock<std::mutex> lock(mImageMutex);
    mCvImageDepth = ImgDepth;
    mbNewDepth = true;
}

} // namespace slam
