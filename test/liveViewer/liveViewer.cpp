#include "oniCamera.h"
#include "mapViewer/mapViewer.h"
#include "fullSystem/fullSystem.h"

int main(int argc, char **argv)
{
    ONICamera camera(640, 480, 30);
    MapViewer viewer(1920, 920);

    Mat33d K;
    K << 525, 0, 320,
        0, 525, 240,
        0, 0, 1;

    Mat depth, image;
    Mat depthFloat, intensity, imageFloat;
    FullSystem fullsystem(640, 480, K, 5, true);

    while (true && !pangolin::ShouldQuit())
    {
        if (camera.getNextImages(depth, image))
        {
            image.convertTo(imageFloat, CV_32FC3);
            cv::cvtColor(imageFloat, intensity, cv::COLOR_RGB2GRAY);
            depth.convertTo(depthFloat, CV_32FC1, 1.0 / 1000);

            viewer.setColourImage(image);
            // viewer.setDepthImage(image);

            if (!viewer.isSystemPaused())
                fullsystem.processFrame(intensity, depthFloat);

            if (viewer.isResetRequested())
                fullsystem.resetSystem();

            viewer.setRawFrameHistory(fullsystem.getRawFramePoseHistory());
        }

        viewer.renderView();
    }

    // ONICamera camera(640, 480, 30);

    // Mat33d K;
    // K << 525, 0, 320,
    //     0, 525, 240,
    //     0, 0, 1;

    // Mat depth, image;
    // Mat depthFloat, intensity, imageFloat;
    // FullSystem fullsystem(640, 480, K, 5, true);

    // while (true)
    // {
    //     if (camera.getNextImages(depth, image))
    //     {
    //         image.convertTo(imageFloat, CV_32FC3);
    //         cv::cvtColor(imageFloat, intensity, cv::COLOR_RGB2GRAY);
    //         depth.convertTo(depthFloat, CV_32FC1, 1.0 / 1000);

    //         fullsystem.processFrame(intensity, depthFloat);
    //     }
    // }
}