#include "FullSystem.h"

int main(int argc, char **argv)
{
    if (argc <= 1)
    {
        std::cout << "usage: ./testRun settings_file vocabulary_file" << std::endl;
        exit(-1);
    }

    FullSystem sys(argv[1], argv[2]);
    cv::Mat imDepth, imRGB;

    while (!sys.IsFinished())
    {
        sys.TrackImageRGBD(imRGB, imDepth, 0);
    }
}