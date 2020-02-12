#include "System.h"
#include "CameraOpenNI.h"

int main(int argc, char **argv)
{
    if (argc <= 2)
    {
        std::cout << "usage: ./liveDemo settingsFile ORBVocFile" << std::endl;
        exit(-1);
    }

    ONI::Camera cam;
    SLAM::System sys(argv[1], argv[2]);

    cv::Mat imDepth, imRGB;

    while (!SLAM::g_bSystemKilled)
    {
        if (cam.TryFetchingImages(imDepth, imRGB))
            sys.trackImage(imRGB, imDepth, 0);
    }
}