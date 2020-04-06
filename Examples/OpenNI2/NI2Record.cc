#include "CoreSystem.h"

int main(int argc, char **argv)
{
    slam::CoreSystem system(argv[1], argv[2]);
    std::string base = "/home/xingrui/Downloads/images3/";
    for (int i = 0; i < 2081; ++i)
    {
        if (slam::g_bSystemKilled)
            return -1;

        if (!slam::g_bSystemRunning)
        {
            i--;
            continue;
        }

        std::stringstream ss1, ss2;
        ss1 << base << i << "_rgb.png";
        ss2 << base << i << "_depth.png";
        cv::Mat im = cv::imread(ss1.str(), cv::IMREAD_UNCHANGED);
        cv::Mat imDepth = cv::imread(ss2.str(), cv::IMREAD_UNCHANGED);
        system.takeNewFrame(im, imDepth, 0);
    }

    base = "/home/xingrui/Downloads/images2/";
    for (int i = 0; i < 2419; ++i)
    {
        if (slam::g_bSystemKilled)
            return -1;

        if (!slam::g_bSystemRunning)
        {
            i--;
            continue;
        }

        std::stringstream ss1, ss2;
        ss1 << base << i << "_rgb.png";
        ss2 << base << i << "_depth.png";
        cv::Mat im = cv::imread(ss1.str(), cv::IMREAD_UNCHANGED);
        cv::Mat imDepth = cv::imread(ss2.str(), cv::IMREAD_UNCHANGED);
        system.takeNewFrame(im, imDepth, 0);
    }

    base = "/home/xingrui/Downloads/images/";
    for (int i = 0; i < 2340; ++i)
    {
        if (slam::g_bSystemKilled)
            return -1;

        if (!slam::g_bSystemRunning)
        {
            i--;
            continue;
        }

        std::stringstream ss1, ss2;
        ss1 << base << i << "_rgb.png";
        ss2 << base << i << "_depth.png";
        cv::Mat im = cv::imread(ss1.str(), cv::IMREAD_UNCHANGED);
        cv::Mat imDepth = cv::imread(ss2.str(), cv::IMREAD_UNCHANGED);
        system.takeNewFrame(im, imDepth, 0);
    }

    while (!slam::g_bSystemKilled)
    {
        usleep(1000);
    }
}