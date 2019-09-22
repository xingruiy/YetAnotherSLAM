#include "oniCamera.h"
#include "mapViewer/mapViewer.h"
#include "fullSystem/fullSystem.h"

int main(int argc, char **argv)
{
    Mat33d K = Mat33d::Identity();
    K(0, 0) = K(1, 1) = 570;
    K(0, 2) = 319.5;
    K(1, 2) = 239.5;
    ONICamera camera(640, 480, 30);
    MapViewer viewer(1920, 920, 640, 480, K);

    Mat depth, image;
    Mat depthFloat, depthImage;
    FullSystem fullsystem(640, 480, K, 5, true);
    float depthScale = 1.0 / 1000.0;

    while (true && !pangolin::ShouldQuit())
    {
        if (camera.getNextImages(depth, image))
        {
            depth.convertTo(depthFloat, CV_32FC1, depthScale);
            depth.convertTo(depthImage, CV_8UC4);
            viewer.setColourImage(image);
            // viewer.setDepthImage(depthImage);

            if (!viewer.paused())
                fullsystem.processFrame(image, depthFloat);

            if (viewer.isResetRequested())
                fullsystem.resetSystem();

            viewer.setRawFrameHistory(fullsystem.getRawFramePoseHistory());
            viewer.setKeyFrameHistory(fullsystem.getKeyFramePoseHistory());
            viewer.setFrameHistory(fullsystem.getFramePoseHistory());
            viewer.setRawKeyFrameHistory(fullsystem.getRawKeyFramePoseHistory());

            if (!viewer.paused())
            {
                float *vbuffer;
                float *nbuffer;
                size_t bufferSize;
                viewer.getMeshBuffer(vbuffer, nbuffer, bufferSize);
                size_t size = fullsystem.getMesh(vbuffer, nbuffer, bufferSize);
                viewer.setMeshSizeToRender(size);
                viewer.setActivePoints(fullsystem.getActiveKeyPoints());
                viewer.setStablePoints(fullsystem.getStableKeyPoints());
            }
        }

        viewer.renderView();
    }
}