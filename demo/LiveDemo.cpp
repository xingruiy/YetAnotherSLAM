#include "FullSystem.h"
#include "ONI_Camera/OniCamera.h"

int main(int argc, char **argv)
{
    if (argc <= 1)
    {
        std::cout << "usage: ./liveDemo settings-file vocabulary-file" << std::endl;
        exit(-1);
    }

    ONI::Camera cam;
    FullSystem sys(argv[1], argv[2]);

    cv::Mat imDepth, imRGB;

    while (!sys.IsFinished())
    {
        if (cam.TryFetchingImages(imDepth, imRGB))
        {
            sys.TrackImageRGBD(imRGB, imDepth, 0);
        }
    }
}