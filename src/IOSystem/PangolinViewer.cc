#include "PangolinViewer.h"
#include "CoreSystem/Frame.h"
#include <pangolin/pangolin.h>

namespace slam
{

PangolinViewer::PangolinViewer() : BaseOutput()
{
    Eigen::Matrix3f K;
    fx = 528;
    fy = 528;
    cx = 320;
    cy = 240;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    Kinv = K.inverse();
}

void PangolinViewer::Run()
{
    pangolin::CreateWindowAndBind("SLAM", 1920, 1080);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto rs = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(width, height, fx, fy, cx, cy, 0.1, 1000),
        pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

    auto MenuDividerLeft = pangolin::Attach::Pix(300);
    float RightSideBarDividerLeft = 0.75f;

    pangolin::CreateDisplay().SetHandler(new pangolin::Handler3D(rs));

    while (!pangolin::ShouldQuit())
    {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        displayLiveCam();

        pangolin::FinishFrame();
    }
}

void PangolinViewer::publishFrame(Frame *F)
{
    std::unique_lock<std::mutex> lock(camPoseMutex);
    camToWorld = F->mTcw.cast<float>().matrix();
}

void PangolinViewer::publishKeyFrame()
{
}

void PangolinViewer::publishGraph()
{
}

void PangolinViewer::displayLiveCam()
{
    pangolin::glDrawFrustum<float>(Kinv, width, height, camToWorld, 0.01f);
}

void PangolinViewer::displayKeyFrame()
{
}

} // namespace slam