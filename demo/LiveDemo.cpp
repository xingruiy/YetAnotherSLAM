#include "System.h"
#include "CameraOpenNI.h"

int main(int argc, char **argv)
{
    if (argc <= 1)
    {
        std::cout << "usage: ./liveDemo settingsFile" << std::endl;
        exit(-1);
    }

    ONI::Camera cam;
    SLAM::System sys(argv[1]);

    cv::Mat imDepth, imRGB;

    while (sys.IsAlive())
    {
        if (cam.TryFetchingImages(imDepth, imRGB))
            sys.TrackImage(imRGB, imDepth, 0);
    }
}