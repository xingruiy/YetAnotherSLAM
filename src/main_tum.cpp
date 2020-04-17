#include "FullSystem.h"
#include "PangolinViewer.h"
#include "GlobalSettings.h"
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace std::chrono;
using namespace cv;

void loadImages(const string &path, vector<string> &rgb_files,
                vector<string> &depth_fiels, vector<double> &time_stamps);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: ./prog settings vocabulary");
        exit(-1);
    }

    vector<string> listOfTests = {
        "/home/xingrui/Downloads/TUM-RGBD/rgbd_dataset_freiburg1_desk2/"};

    printf("==== initializing the system\n");
    slam::FullSystem slam(argv[1], argv[2]);
    slam::PangolinViewer viewer(1920, 1080);
    slam.addOutput(&viewer);

    const int totalIter = 1;
    for (auto test : listOfTests)
    {
        vector<string> rgb_files;
        vector<string> depth_files;
        vector<double> time_stamps;
        loadImages(test, rgb_files, depth_files, time_stamps);

        const int nImages = rgb_files.size();
        printf("==== starting testing on dataset %s with %d images\n", test.c_str(), nImages);

        string result_dir = test + "results/";
        string cmd = "mkdir -p " + result_dir;

        RUNTIME_ASSERT(rgb_files.size() == depth_files.size());
        RUNTIME_ASSERT(time_stamps.size() == rgb_files.size());
        RUNTIME_ASSERT(0 == system(cmd.c_str()))

        for (int iter = 0; iter < totalIter; ++iter)
        {
            slam.reset();

            for (int i = 0; i < nImages; ++i)
            {
                if (slam.isShutdown())
                    return -1;

                auto t1 = steady_clock::now();
                Mat rgb = imread(test + rgb_files[i], CV_LOAD_IMAGE_UNCHANGED);
                Mat depth = imread(test + depth_files[i], CV_LOAD_IMAGE_UNCHANGED);

                Mat rgbGray;
                Mat depthFloat;

                cvtColor(rgb, rgbGray, COLOR_BGR2GRAY);
                depth.convertTo(depthFloat, CV_32FC1, 1.0 / 5000);

                const double ts = time_stamps[i];
                slam.addImages(rgbGray, depthFloat, ts);
                auto t2 = steady_clock::now();
                double ttrack = duration_cast<duration<double>>(t2 - t1).count();

                double T = 0;
                if (i < nImages - 1)
                    T = time_stamps[i + 1] - ts;
                else if (i > 0)
                    T = ts - time_stamps[i - 1];

                if (ttrack < T)
                    usleep((T - ttrack) * 1e6);
            }

            usleep(10 * 1e6);
            stringstream ss, ss2;
            ss << result_dir << iter << "th_run.txt";
            ss2 << result_dir << iter << "th_run_kf.txt";
            slam.SaveTrajectoryTUM(ss.str());
            slam.SaveKeyFrameTrajectoryTUM(ss2.str());
        }
    }
}

void loadImages(const string &path, vector<string> &rgb_files,
                vector<string> &depth_fiels, vector<double> &time_stamps)
{
    ifstream file;
    file.open((path + "associated.txt").c_str());
    while (!file.eof())
    {
        string s;
        getline(file, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            time_stamps.push_back(t);
            ss >> sRGB;
            rgb_files.push_back(sRGB);
            ss >> t;
            ss >> sD;
            depth_fiels.push_back(sD);
        }
    }
}
