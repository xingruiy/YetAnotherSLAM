#include <pangolin/pangolin.h>
#include "utils/numType.h"

void drawCall0()
{
    Mat33d K, Kinv;
    K << 570, 0, 319.5,
        0, 570, 239.5,
        0, 0, 1;
    Kinv = K.inverse();

    Vec3d ptBefore = {-0.114898, 0.0440001, 0.847969};
    Vec3d ptAfter = {-0.134624, 0.0569476, 1.27336};
    Mat44d camAPose = Mat44d::Identity(), camBPose = Mat44d::Identity();
    camAPose << 0.955103, 0.172899, -0.24059, -0.0544899, -0.105483, 0.957292, 0.26194, 0.00273962, 0.276858, -0.23173, 0.932551, -0.187454;
    camBPose << 0.919124, 0.242461, -0.310521, -0.0671703, -0.159894, 0.949931, 0.26848, 0.00979193, 0.360062, -0.197086, 0.911873, -0.208095;
    Vec2d projA = {450.884, 118.232};
    Vec2d projB = {510.603, 131.383};
    Vec3d camACenter = camAPose.topRightCorner(3, 1);
    Vec3d camBCenter = camBPose.topRightCorner(3, 1);
    Vec3d ptA = camAPose.topLeftCorner(3, 3) * Vec3d((projA(0) - 319.5) / 570.0 * 3.0, (projA(1) - 239.5) / 570.0 * 3.0, 3.0) + camAPose.topRightCorner(3, 1);
    Vec3d ptB = camBPose.topLeftCorner(3, 3) * Vec3d((projB(0) - 319.5) / 570.0 * 3.0, (projB(1) - 239.5) / 570.0 * 3.0, 3.0) + camBPose.topRightCorner(3, 1);
    std::vector<Vec3d> lines = {camACenter, ptA, camBCenter, ptB};
    std::vector<Vec3d> pointsA = {camACenter, camBCenter, ptBefore};
    std::vector<Vec3d> pointsB = {ptAfter};
    pangolin::glDrawLines(lines);
    glPointSize(15.0f);
    pangolin::glDrawPoints(pointsA);
    glColor4f(1.0, 0, 0, 1.0);
    pangolin::glDrawFrustum<float>(Kinv.cast<float>(), 640, 480, camAPose.cast<float>(), 0.1f);
    pangolin::glDrawFrustum<float>(Kinv.cast<float>(), 640, 480, camBPose.cast<float>(), 0.1f);
    glColor4f(1.0, 0, 1.0, 1.0);
    pangolin::glDrawPoints(pointsB);

    glColor4f(1.0, 1.0, 1.0, 1.0);
}

int main(int argc, char **argv)
{
    pangolin::CreateWindowAndBind("yes");

    auto mainCamera = std::make_shared<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);

    auto modelView = &pangolin::Display("Viewer");
    modelView->SetHandler(new pangolin::Handler3D(*mainCamera));

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.f, 0.f, 0.f, 1.f);

        modelView->Activate(*mainCamera);
        drawCall0();
        pangolin::FinishFrame();
    }
}