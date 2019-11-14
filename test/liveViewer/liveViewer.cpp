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
    FullSystem fullsystem(640, 480, K, 5, viewer);
    Mat depth, image;
    Mat depthFloat, depthImage;

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

            if (viewer.displayImageDepth())
            {
                computeVMap(gpuBufferFloatWxH, bufferVec4FloatWxH, K);
                computeNormal(bufferVec4FloatWxH, bufferVec4FloatWxH2);
                renderScene(bufferVec4FloatWxH, bufferVec4FloatWxH2, bufferVec4ByteWxH);
                viewer.setDepthImage(Mat(bufferVec4ByteWxH));
            }

            if (viewer.displayImageRGB())
                viewer.setColourImage(image);

            if (viewer.isLocalizationMode())
                fullsystem.setSystemStateToLost();

            if (viewer.isDebugRequested())
                fullsystem.setSystemStateToTest();

            if (viewer.isNextKFRequested())
                fullsystem.testNextKF();

            if (viewer.isResetRequested())
                fullsystem.resetSystem();

            fullsystem.setMappingEnable(viewer.mappingEnabled());
            fullsystem.setGraphMatching(viewer.isGraphMatchingMode());
            fullsystem.setGraphGetNormal(viewer.shouldCalculateNormal());

            if (!viewer.paused())
            {
                fullsystem.setCpuBufferVec4FloatWxH(Mat(bufferVec4FloatWxH2));
                fullsystem.processFrame(image, depthFloat);
            }

            if (!viewer.paused())
            {
                float *vbuffer;
                float *nbuffer;
                size_t bufferSize;
                viewer.getMeshBuffer(vbuffer, nbuffer, bufferSize);
                viewer.setMeshSizeToRender(fullsystem.getMesh(vbuffer, nbuffer, bufferSize));
                viewer.setActivePoints(fullsystem.getMapPointPosAll());
                // viewer.setKeyFrameHistory(fullsystem.getKeyFramePoseHistory());
                // viewer.setFrameHistory(fullsystem.getFramePoseHistory());
            }
        }

        viewer.renderView();
    }
}