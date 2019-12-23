#include "FullSystem.h"
#include "OniCamera/OniCamera.h"

int main(int argc, char **argv)
{
    if (argc <= 1)
    {
        exit(-1);
    }

    FullSystem sys(argv[1], argv[2]);
    ONI::Camera cam;

    cv::Mat imDepth, imRGB;

    while (!sys.IsFinished())
    {
        if (cam.TryFetchingImages(imDepth, imRGB))
        {
            sys.TrackImageRGBD(imRGB, imDepth, 0);
        }
    }
}