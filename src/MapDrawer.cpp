#include "MapDrawer.h"

namespace SLAM
{

MapDrawer::MapDrawer(Map *pMap) : mpMap(pMap)
{
    calibInv = g_calibInv[0];
    width = g_width[0];
    height = g_height[0];
}

void MapDrawer::DrawKeyFrames(bool bDrawKF, bool bDrawGraph, int N)
{
    const auto vpKFs = mpMap->GetAllKeyFrames();

    if (bDrawKF)
    {
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            Eigen::Matrix4f Tcw = pKF->mTcw.matrix().cast<float>();

            glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
            pangolin::glDrawFrustum(calibInv, width, height, Tcw, 0.05f);
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        }
    }

    if (bDrawGraph)
    {
        glLineWidth(1);
        glColor4f(0.5f, 1.0f, 0.0f, 1.0f);
        glBegin(GL_LINES);

        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            // Covisibility Graph
            const auto vCovKFs = vpKFs[i]->GetBestCovisibilityKeyFrames(N);
            Eigen::Vector3f Ow = vpKFs[i]->mTcw.translation().cast<float>();
            if (!vCovKFs.empty())
            {
                for (auto vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
                {
                    if ((*vit)->mnId < vpKFs[i]->mnId)
                        continue;

                    Eigen::Vector3f Ow2 = (*vit)->mTcw.translation().cast<float>();
                    glVertex3f(Ow(0), Ow(1), Ow(2));
                    glVertex3f(Ow2(0), Ow2(1), Ow2(2));
                }
            }

            // Spanning tree
            // KeyFrame *pParent = vpKFs[i]->GetParent();
            // if (pParent)
            // {
            //     Eigen::Vector3f Owp = pParent->mTcw.translation().cast<float>();
            //     glVertex3f(Ow(0), Ow(1), Ow(2));
            //     glVertex3f(Owp(0), Owp(1), Owp(2));
            // }

            // Loops
            // set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();
            // for (set<KeyFrame *>::iterator sit = sLoopKFs.begin(), send = sLoopKFs.end(); sit != send; sit++)
            // {
            //     if ((*sit)->mnId < vpKFs[i]->mnId)
            //         continue;
            //     cv::Mat Owl = (*sit)->GetCameraCenter();
            //     glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
            //     glVertex3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2));
            // }
        }

        glEnd();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

void MapDrawer::DrawMapPoints(int iPointSize)
{
    const auto &vpMPs = mpMap->GetAllMapPoints();
    const auto &vpRefMPs = mpMap->GetReferenceMapPoints();

    std::set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if (vpMPs.empty())
        return;

    glPointSize(iPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;

        Eigen::Vector3f pos = vpMPs[i]->mWorldPos.cast<float>();
        glVertex3f(pos(0), pos(1), pos(2));
    }
    glEnd();

    glPointSize(iPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);

    for (set<MapPoint *>::iterator sit = spRefMPs.begin(), send = spRefMPs.end(); sit != send; sit++)
    {
        if ((*sit)->isBad())
            continue;

        Eigen::Vector3f pos = (*sit)->mWorldPos.cast<float>();
        glVertex3f(pos(0), pos(1), pos(2));
    }

    glEnd();
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
}

} // namespace SLAM