#include "FullSystem/CoreSystem.h"
#include "OpenNI/include/Camera.h"

int main(int argc, char **argv)
{
    if (argc <= 2)
    {
        std::cout << "usage: ./liveDemo settingsFile ORBVocFile" << std::endl;
        exit(-1);
    }

    OpenNI2::Camera cam;
    SLAM::CoreSystem system(argv[1], argv[2]);

    cv::Mat imDepth, imRGB;

    while (!SLAM::g_bSystemKilled)
    {
        if (cam.TryFetchingImages(imDepth, imRGB))
            system.TrackRGBD(imRGB, imDepth, 0);
    }
}