#include "PangolinViewer.h"
#include <thread>
#include "Map.h"
#include "Frame.h"
#include "FullSystem.h"
#include "MapPoint.h"

namespace slam
{

using namespace std;
using namespace Eigen;

PangolinViewer::PangolinViewer(int w, int h) : BaseIOWrapper(), map(0), fsIO(0), w(w), h(h)
{
    new thread(&PangolinViewer::run, this);
}

void PangolinViewer::run()
{
    pangolin::CreateWindowAndBind("SLAM: Viewer", w, h);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto rs = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(640, 480, 528, 528, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

    auto view = &pangolin::Display("map")
                     .SetBounds(0, 1, pangolin::Attach::Pix(300), 1)
                     .SetHandler(new pangolin::Handler3D(rs));

    pangolin::CreatePanel("menu").SetBounds(0, 1, 0, pangolin::Attach::Pix(300));
    pangolin::Var<bool> reset_btn("menu.RESET", false, false);
    pangolin::Var<bool> display_points("menu.Display Point", true, true);
    pangolin::Var<bool> display_graph("menu.Display Graph", true, true);
    pangolin::Var<bool> display_kf("menu.Display KeyFrame", true, true);

    pangolin::Var<int> size_bar("menu.Point Size", 2, 1, 10);

    while (!pangolin::ShouldQuit())
    {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (display_points)
        {
            view->Activate(rs);
            drawMapPoints(size_bar);
        }

        drawLiveFrame(0.05);
        drawKeyFrames(display_graph, display_kf, 0);
        pangolin::FinishFrame();
    }

    if (fsIO)
        fsIO->shutdown();
}

void PangolinViewer::drawKeyFrames(bool drawGraph, bool drawKF, int N)
{
    if (!map)
        return;

    // const auto vpKFs = mpMap->GetAllKeyFrames();

    // if (bDrawKF)
    // {
    //     for (size_t i = 0; i < vpKFs.size(); i++)
    //     {
    //         KeyFrame *pKF = vpKFs[i];
    //         Matrix4f Tcw = pKF->GetPose().matrix().cast<float>();

    //         glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
    //         pangolin::glDrawFrustum(mCalibInv, width, height, Tcw, 0.05f);
    //         glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    //     }
    // }

    // if (bDrawGraph)
    // {
    //     glLineWidth(1);
    //     glBegin(GL_LINES);

    //     for (size_t i = 0; i < vpKFs.size(); i++)
    //     {
    //         // Covisibility Graph
    //         glColor4f(0.5f, 1.0f, 0.0f, 1.0f);
    //         const auto vCovKFs = vpKFs[i]->GetBestCovisibilityKeyFrames(N);
    //         Vector3f Ow = vpKFs[i]->GetTranslation().cast<float>();
    //         if (!vCovKFs.empty())
    //         {
    //             for (auto vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
    //             {
    //                 if ((*vit)->mnId < vpKFs[i]->mnId)
    //                     continue;

    //                 Vector3f Ow2 = (*vit)->GetTranslation().cast<float>();
    //                 glVertex3f(Ow(0), Ow(1), Ow(2));
    //                 glVertex3f(Ow2(0), Ow2(1), Ow2(2));
    //             }
    //         }

    //         // Loops edge
    //         glColor4f(0.0f, 0.5f, 1.0f, 1.0f);
    //         set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();
    //         for (auto sit = sLoopKFs.begin(), send = sLoopKFs.end(); sit != send; sit++)
    //         {
    //             if ((*sit)->mnId < vpKFs[i]->mnId)
    //                 continue;
    //             Vector3f Owl = (*sit)->GetTranslation().cast<float>();
    //             glVertex3f(Ow(0), Ow(1), Ow(2));
    //             glVertex3f(Owl(0), Owl(1), Owl(2));
    //         }
    //     }

    //     glEnd();
    //     glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    // }
}

void PangolinViewer::drawMapPoints(int size)
{
    if (!map)
        return;

    const auto &vpMPs = map->GetAllMapPoints();
    const auto &vpRefMPs = map->GetReferenceMapPoints();
    set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if (vpMPs.empty())
        return;

    glPointSize(size);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;

        Vector3f pos = vpMPs[i]->mWorldPos.cast<float>();
        glVertex3f(pos(0), pos(1), pos(2));
    }

    glEnd();
    glPointSize(size);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);

    for (MapPoint *pMP : spRefMPs)
        if (!pMP->isBad())
        {
            Vector3f pos = pMP->mWorldPos.cast<float>();
            glVertex3f(pos(0), pos(1), pos(2));
        }

    glEnd();
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
}

void PangolinViewer::drawLiveFrame(float scale)
{
    Matrix3f K;
    K << 528, 0, 320, 0, 528, 240, 0, 0, 1;
    pangolin::glDrawFrustum<float>(K.inverse(), 640, 480, liveFramePose, scale);
}

void PangolinViewer::setGlobalMap(Map *map)
{
    this->map = map;
}

void PangolinViewer::setSystemIO(FullSystem *fs)
{
    fsIO = fs;
}

void PangolinViewer::publishLiveFrame(Frame *newF)
{
    liveFramePose = newF->mTcw.matrix().cast<float>();
    cout << "==== live frame pose:" << endl
         << liveFramePose << endl;
}

} // namespace slam
