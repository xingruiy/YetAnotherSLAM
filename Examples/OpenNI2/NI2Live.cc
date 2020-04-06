#include "CoreSystem.h"
#include "utils/GlobalSettings.h"
#include "OpenNI/include/Camera.h"

int main(int argc, char **argv)
{
    if (argc <= 2)
    {
        std::cout << "usage: ./liveDemo settingsFile ORBVocFile" << std::endl;
        exit(-1);
    }

    OpenNI2::Camera cam;
    slam::GlobalSettings settings(argv[1]);
    slam::CoreSystem system(&settings, argv[2]);

    cv::Mat imDepth, imRGB;

    while (!slam::g_bSystemKilled)
    {
        if (cam.TryFetchingImages(imDepth, imRGB))
            system.takeNewFrame(imRGB, imDepth, 0);
    }
}