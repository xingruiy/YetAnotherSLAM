#include "System.h"
#include "Camera.h"

int main(int argc, char **argv)
{
    if (argc <= 2)
    {
        std::cout << "usage: ./liveDemo settingsFile ORBVocFile" << std::endl;
        exit(-1);
    }

    OpenNI2::Camera cam;
    SLAM::System system(argv[1], argv[2]);

    cv::Mat imDepth, imRGB;

    while (!SLAM::g_bSystemKilled)
    {
        if (cam.TryFetchingImages(imDepth, imRGB))
            system.TrackRGBD(imRGB, imDepth, 0);
    }

    // SLAM::System system(argv[1], argv[2]);
    // std::string base = "/home/xyang/images/";
    // for (int i = 0; i < 2487; ++i)
    // {
    //     if (!SLAM::g_bSystemRunning)
    //     {
    //         i--;
    //         continue;
    //     }

    //     std::stringstream ss1, ss2;
    //     ss1 << base << i << "_rgb.png";
    //     ss2 << base << i << "_depth.png";
    //     cv::Mat im = cv::imread(ss1.str(), cv::IMREAD_UNCHANGED);
    //     cv::Mat imDepth = cv::imread(ss2.str(), cv::IMREAD_UNCHANGED);
    //     system.TrackRGBD(im, imDepth, 0);
    // }

    // while (!SLAM::g_bSystemKilled)
    // {
    //     usleep(1000);
    // }
}