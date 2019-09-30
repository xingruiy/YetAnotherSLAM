#include "tumLoader.h"
#include "mapViewer/mapViewer.h"
#include "fullSystem/fullSystem.h"

int main(int argc, char **argv)
{
    Mat33d K = Mat33d::Identity();
    K(0, 0) = K(1, 1) = 525.0;
    K(0, 2) = 319.5;
    K(1, 2) = 239.5;

    MapViewer viewer(1920, 920, 640, 480, K);
    std::string baseDir = "/home/xyang/Downloads/TUM-RGBD/";
    std::vector<std::string> listOfFilePath = {
        // "rgbd_dataset_freiburg1_xyz/",
        // "rgbd_dataset_freiburg1_rpy/",
        // "rgbd_dataset_freiburg2_xyz/",
        // "rgbd_dataset_freiburg2_rpy/",
        "rgbd_dataset_freiburg3_long_office_household/",
        // "rgbd_dataset_freiburg3_structure_texture_far/",
        // "rgbd_dataset_freiburg3_structure_texture_near/",
        ""};

    for (auto iter = listOfFilePath.begin(); iter != listOfFilePath.end() && *iter != ""; ++iter)
    {
        Mat depth, image;
        Mat depthFloat;
        viewer.resetViewer();
        FullSystem fullsystem(640, 480, K, 5, true);
        fullsystem.setMapViewerPtr(&viewer);

        printf("Trying: %s...\n", iter->c_str());

        std::vector<double> listOfTimeStamp;
        std::vector<std::string> listOfDepthPath;
        std::vector<std::string> listOfImagePath;
        std::vector<SE3> trajectory;

        TUMLoad(baseDir + *iter, listOfTimeStamp, listOfDepthPath, listOfImagePath);
        printf("A total number of \033[1;31m%lu\033[0m Images to be processed\n", listOfTimeStamp.size());

        for (int i = 0; i < listOfTimeStamp.size(); ++i)
        {
            // printf("processing: %i / %lu \r", i, listOfTimeStamp.size());
            depth = cv::imread(listOfDepthPath[i], cv::IMREAD_UNCHANGED);
            image = cv::imread(listOfImagePath[i], cv::IMREAD_UNCHANGED);
            depth.convertTo(depthFloat, CV_32FC1, 1 / 5000.f);

            viewer.setColourImage(image);
            // viewer.setDepthImage(image);

            fullsystem.processFrame(image, depthFloat);

            if (viewer.isResetRequested())
                fullsystem.resetSystem();

            viewer.setKeyFrameHistory(fullsystem.getKeyFramePoseHistory());
            viewer.setFrameHistory(fullsystem.getFramePoseHistory());

            float *vbuffer;
            float *nbuffer;
            size_t bufferSize;
            viewer.getMeshBuffer(vbuffer, nbuffer, bufferSize);
            size_t size = fullsystem.getMesh(vbuffer, nbuffer, bufferSize);
            viewer.setMeshSizeToRender(size);
            viewer.setActivePoints(fullsystem.getMapPointPosAll());

            if (pangolin::ShouldQuit())
                break;

            viewer.renderView();
        }

        while (!pangolin::ShouldQuit())
        {
            viewer.renderView();
            viewer.setKeyFrameHistory(fullsystem.getKeyFramePoseHistory());
            viewer.setFrameHistory(fullsystem.getFramePoseHistory());

            float *vbuffer;
            float *nbuffer;
            size_t bufferSize;
            viewer.getMeshBuffer(vbuffer, nbuffer, bufferSize);
            size_t size = fullsystem.getMesh(vbuffer, nbuffer, bufferSize);
            viewer.setMeshSizeToRender(size);
            viewer.setActivePoints(fullsystem.getMapPointPosAll());
        }

        if (fullsystem.getFramePoseHistory().size() == listOfFilePath.size())
        {
            TUMSave(baseDir + *iter, listOfTimeStamp, fullsystem.getFramePoseHistory());
            printf("Saved: %s...\n", iter->c_str());
        }
    }
}