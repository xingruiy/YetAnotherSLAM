#include "fullSystem/fullSystem.h"
#include "mapViewer/mapViewer.h"
#include "oniCamera.h"
#include <thread>

MapViewer *viewer;

static void runViewer()
{
    while (!pangolin::ShouldQuit())
    {
        if (viewer != NULL)
            viewer->renderView();
    }
}

int main(int argc, char **argv)
{
    viewer = new MapViewer(1280, 900);
    ONICamera camera(640, 480, 30);

    Mat33d K;
    K << 525, 0, 320,
        0, 525, 240,
        0, 0, 1;

    auto t = std::thread(&runViewer);
    while (true && !pangolin::ShouldQuit())
    {
        Mat depth, image;
        FullSystem fullsystem(640, 480, K, 5, true);
        if (camera.getNextImages(depth, image))
        {
            viewer->setColourImage(image);
            viewer->setDepthImage(image);

            fullsystem.processFrame(image, depth);

            if (viewer->isResetRequested())
                fullsystem.resetSystem();
        }
    }

    t.join();
}