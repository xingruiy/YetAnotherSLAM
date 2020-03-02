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
    SLAM::System sys(argv[1], argv[2]);

    cv::Mat imDepth, imRGB;

    while (!SLAM::g_bSystemKilled)
    {
        if (cam.TryFetchingImages(imDepth, imRGB))
            sys.TrackRGBD(imRGB, imDepth, 0);
    }

    // OpenNI2::Camera cam;
    // SLAM::System sys(argv[1], argv[2]);

    // cv::Mat imDepth, imRGB;
    // size_t ImageId = 0;

    // while (!SLAM::g_bSystemKilled)
    // {
    //     if (cam.TryFetchingImages(imDepth, imRGB))
    //     {
    //         // sys.TrackRGBD(imRGB, imDepth, 0);
    //         cv::imshow("depth", imDepth);
    //         cv::imshow("rgb", imRGB);
    //         cv::waitKey(1);

    //         if (SLAM::g_bSystemRunning && !imDepth.empty() && !imRGB.empty())
    //         {
    //             std::stringstream ss1, ss2;
    //             ss1 << "home/xyang/images/depth_" << ImageId << ".bmp";
    //             ss2 << "home/xyang/images/rgb_" << ImageId << ".bmp";
    //             std::cout << ss1.str() << std::endl;
    //             cv::imwrite(ss1.str(), imDepth);
    //             cv::imwrite(ss2.str(), imRGB);

    //             std::cout << "images saved: " << ImageId << std::endl;
    //             ImageId++;
    //         }
    //     }
    // }
}