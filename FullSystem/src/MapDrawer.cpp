#include "MapDrawer.h"

namespace SLAM
{

MapDrawer::MapDrawer(Map *pMap) : mpMap(pMap)
{
    mCalibInv = g_calibInv[0];
    width = g_width[0];
    height = g_height[0];
}

void MapDrawer::LinkGlSlProgram()
{
    const char vertexShader[] =
        "#version 330\n"
        "\n"
        "layout(location = 0) in vec3 position;\n"
        "layout(location = 1) in vec3 a_normal;\n"
        "uniform mat4 mvpMat;\n"
        "uniform mat4 Tmw;\n"
        "uniform float colourTaint;\n"
        "out vec3 shaded_colour;\n"
        "\n"
        "void main(void) {\n"
        "    gl_Position = mvpMat * Tmw * vec4(position, 1.0);\n"
        "    vec3 lightpos = vec3(5, 5, 5);\n"
        "    const float ka = 0.3;\n"
        "    const float kd = 0.5;\n"
        "    const float ks = 0.2;\n"
        "    const float n = 20.0;\n"
        "    float ax = 1.0;\n"
        "    float dx = 1.0;\n"
        "    float sx = 1.0;\n"
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

    mShader.AddShader(pangolin::GlSlVertexShader, vertexShader);
    mShader.AddShader(pangolin::GlSlFragmentShader, fragShader);
    mShader.Link();
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
            pangolin::glDrawFrustum(mCalibInv, width, height, Tcw, 0.05f);
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

                    Eigen::Vector3f Ow2 = (*vit)->GetTranslation().cast<float>();
                    glVertex3f(Ow(0), Ow(1), Ow(2));
                    glVertex3f(Ow2(0), Ow2(1), Ow2(2));
                }
            }

            // Loops
            std::set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for (auto sit = sLoopKFs.begin(), send = sLoopKFs.end(); sit != send; sit++)
            {
                if ((*sit)->mnId < vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetTranslation().cast<float>();
                glVertex3f(Ow(0), Ow(1), Ow(2));
                glVertex3f(Owl(0), Owl(1), Owl(2));
            }
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

    for (std::set<MapPoint *>::iterator sit = spRefMPs.begin(), send = spRefMPs.end(); sit != send; sit++)
    {
        if ((*sit)->isBad())
            continue;

        Eigen::Vector3f pos = (*sit)->mWorldPos.cast<float>();
        glVertex3f(pos(0), pos(1), pos(2));
    }

    glEnd();
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
}

void MapDrawer::DrawMesh(int N, const pangolin::OpenGlMatrix &mvpMat)
{
    std::vector<MapStruct *> vpMapStruct = mpMap->GetAllVoxelMaps();
    if (N <= 0 || vpMapStruct.size() < N)
        N = vpMapStruct.size();
    std::vector<MapStruct *> vpMSToDraw = std::vector<MapStruct *>(vpMapStruct.end() - N, vpMapStruct.end());

    for (auto vit = vpMSToDraw.begin(), vend = vpMSToDraw.end(); vit != vend; ++vit)
    {
        MapStruct *pMS = *vit;
        if (pMS && !pMS->mbActive)
        {
            if (pMS->mbInHibernation)
            {
                if (pMS->mbVertexBufferCreated)
                {
                    glDeleteBuffers(1, &pMS->mGlVertexBuffer);
                    glDeleteBuffers(1, &pMS->mGlNormalBuffer);
                    pMS->mbVertexBufferCreated = false;
                }

                continue;
            }

            if (!pMS->mbHasMesh)
                pMS->GenerateMesh();

            if (!pMS->mplPoint)
                return;

            if (!pMS->mbVertexBufferCreated)
            {
                glGenBuffers(1, &pMS->mGlVertexBuffer);
                glBindBuffer(GL_ARRAY_BUFFER, pMS->mGlVertexBuffer);
                glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pMS->N, pMS->mplPoint, GL_STATIC_DRAW);
                glGenBuffers(1, &pMS->mGlNormalBuffer);
                glBindBuffer(GL_ARRAY_BUFFER, pMS->mGlNormalBuffer);
                glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pMS->N, pMS->mplNormal, GL_STATIC_DRAW);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                pMS->mbVertexBufferCreated = true;
            }

            Eigen::Matrix4f Tmw = pMS->mTcw.matrix().cast<float>();

            mShader.Bind();
            mShader.SetUniform("Tmw", pangolin::OpenGlMatrix(Tmw));
            mShader.SetUniform("mvpMat", mvpMat);
            mShader.SetUniform("colourTaint", pMS->mColourTaint);

            glEnableVertexAttribArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, pMS->mGlVertexBuffer);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, pMS->mGlNormalBuffer);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

            glDrawArrays(GL_TRIANGLES, 0, pMS->N);
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            mShader.Unbind();
        }
    }
}

} // namespace SLAM