#include "FullSystem.h"
#include "GlobalSettings.h"
#include <fstream>
#include <opencv2/opencv.hpp>

void loadImages(
    const std::string &path,
    std::vector<std::string> &rgb_files,
    std::vector<std::string> &depth_fiels,
    std::vector<double> &time_stamps)
{
    std::ifstream fAssociation;
    fAssociation.open((path + "associated.txt").c_str());
    while (!fAssociation.eof())
    {
        std::string s;
        getline(fAssociation, s);
        if (!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double t;
            std::string sRGB, sD;
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

inline bool file_exist(const std::string &filename)
{
    std::ifstream file(filename.c_str());
    return file.good();
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: ./prog settings vocabulary");
        exit(-1);
    }

    std::vector<std::string> listOfTests = {
        "/home/xingrui/Downloads/TUM-RGBD/rgbd_dataset_freiburg1_desk2/"};

    printf("==== initializing the system\n");
    slam::FullSystem slam(argv[1], argv[2]);

    const int totalIter = 1;
    for (auto test : listOfTests)
    {
        std::vector<std::string> rgb_files;
        std::vector<std::string> depth_files;
        std::vector<double> time_stamps;
        loadImages(test, rgb_files, depth_files, time_stamps);

        RUNTIME_ASSERT(rgb_files.size() == depth_files.size());
        RUNTIME_ASSERT(time_stamps.size() == rgb_files.size());

        const int nImages = rgb_files.size();
        printf("==== starting testing on dataset %s with %d images\n", test.c_str(), nImages);

        std::string result_dir = test + "results/";
        std::string cmd = "mkdir -p " + result_dir;
        RUNTIME_ASSERT(0 == system(cmd.c_str()))

        for (int iter = 0; iter < totalIter; ++iter)
        {
            slam.reset();

            for (int i = 0; i < nImages; ++i)
            {
                auto t1 = std::chrono::steady_clock::now();
                cv::Mat rgb = cv::imread(test + rgb_files[i], CV_LOAD_IMAGE_UNCHANGED);
                cv::Mat depth = cv::imread(test + depth_files[i], CV_LOAD_IMAGE_UNCHANGED);

                cv::Mat rgbGray;
                cv::Mat depthFloat;

                cv::cvtColor(rgb, rgbGray, cv::COLOR_BGR2GRAY);
                depth.convertTo(depthFloat, CV_32FC1, 1.0 / 5000);

                const double ts = time_stamps[i];
                slam.addImages(rgbGray, depthFloat, ts);
                auto t2 = std::chrono::steady_clock::now();
                double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

                double T = 0;
                if (i < nImages - 1)
                    T = time_stamps[i + 1] - ts;
                else if (i > 0)
                    T = ts - time_stamps[i - 1];

                if (ttrack < T)
                    usleep((T - ttrack) * 1e6);
            }

            usleep(10 * 1e6);
            std::stringstream ss, ss2;
            ss << result_dir << iter << "th_run.txt";
            ss2 << result_dir << iter << "th_run_kf.txt";
            slam.SaveTrajectoryTUM(ss.str());
            slam.SaveKeyFrameTrajectoryTUM(ss2.str());
        }
    }

    printf("all done!\n");
}
