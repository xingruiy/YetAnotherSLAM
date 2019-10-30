#include "oniCamera.h"
#include "mapViewer/mapViewer.h"
#include "fullSystem/fullSystem.h"
#include "denseTracker/cudaImageProc.h"

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
    fullsystem.setMapViewerPtr(&viewer);
    float depthScale = 1.0 / 1000.0;

    GMat gpuBufferFloatWxH;
    GMat bufferVec4FloatWxH;
    GMat bufferVec4FloatWxH2;
    GMat bufferVec4ByteWxH;

    while (true && !pangolin::ShouldQuit())
    {
        if (camera.getNextImages(depth, image))
        {
            depth.convertTo(depthFloat, CV_32FC1, depthScale);
            gpuBufferFloatWxH.upload(depthFloat);
            computeVMap(gpuBufferFloatWxH, bufferVec4FloatWxH, K);
            computeNormal(bufferVec4FloatWxH, bufferVec4FloatWxH2);
            renderScene(bufferVec4FloatWxH, bufferVec4FloatWxH2, bufferVec4ByteWxH);

            viewer.setColourImage(image);
            viewer.setDepthImage(Mat(bufferVec4ByteWxH));

            fullsystem.setMappingEnable(viewer.mappingEnabled());

            if (viewer.isLocalizationMode())
            {
                fullsystem.setSystemStateToLost();
                fullsystem.setGraphMatching(viewer.isGraphMatchingMode());
                fullsystem.setGraphGetNormal(viewer.shouldCalculateNormal());
            }

            if (!viewer.paused())
            {
                fullsystem.setCurrentNormal(bufferVec4FloatWxH2);
                fullsystem.processFrame(image, depthFloat);
            }

            if (viewer.isResetRequested())
                fullsystem.resetSystem();

            if (!viewer.paused() && viewer.mappingEnabled())
            {
                float *vbuffer;
                float *nbuffer;
                size_t bufferSize;
                viewer.getMeshBuffer(vbuffer, nbuffer, bufferSize);
                size_t size = fullsystem.getMesh(vbuffer, nbuffer, bufferSize);
                viewer.setMeshSizeToRender(size);
                viewer.setActivePoints(fullsystem.getMapPointPosAll());
                viewer.setKeyFrameHistory(fullsystem.getKeyFramePoseHistory());
                viewer.setFrameHistory(fullsystem.getFramePoseHistory());
            }
        }

        viewer.renderView();
    }
}