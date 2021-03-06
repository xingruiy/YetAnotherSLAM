#include "System.h"

int main(int argc, char **argv)
{
    SLAM::System system(argv[1], argv[2]);
    std::string base = "/home/xyang/images/";
    for (int i = 0; i < 2487; ++i)
    {
        if (SLAM::g_bSystemKilled)
            return -1;

        if (!SLAM::g_bSystemRunning)
        {
            i--;
            continue;
        }

        std::stringstream ss1, ss2;
        ss1 << base << i << "_rgb.png";
        ss2 << base << i << "_depth.png";
        cv::Mat im = cv::imread(ss1.str(), cv::IMREAD_UNCHANGED);
        cv::Mat imDepth = cv::imread(ss2.str(), cv::IMREAD_UNCHANGED);
        system.TrackRGBD(im, imDepth, 0);
    }

    while (!SLAM::g_bSystemKilled)
    {
        usleep(1000);
    }
}