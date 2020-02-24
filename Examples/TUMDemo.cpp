#include "System.h"

void LoadImages(const std::string &DatasetRootPath,
                std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD,
                std::vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if (argc <= 3)
    {
        std::cout << "usage: ./liveDemo settingsFile ORBVocFile ImagesRoot" << std::endl;
        exit(-1);
    }

    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestamps;
    std::string ImagesRoot = std::string(argv[3]);
    LoadImages(ImagesRoot, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    cv::Mat imDepth, imRGB;
    size_t nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty())
    {
        std::cerr << endl
                  << "No images found in provided path." << std::endl;
        return 1;
    }
    else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        std::cerr << endl
                  << "Different number of images for rgb and depth." << std::endl;
        return 1;
    }

    std::cout << std::endl
              << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl
              << std::endl;

    SLAM::System sys(argv[1], argv[2]);

    for (int i = 0; i < nImages; ++i)
    {
        if (SLAM::g_bSystemKilled)
            break;

        if (SLAM::g_bSystemRunning)
        {
            imRGB = cv::imread(std::string(argv[3]) + "/" + vstrImageFilenamesRGB[i], CV_LOAD_IMAGE_UNCHANGED);
            imDepth = cv::imread(std::string(argv[3]) + "/" + vstrImageFilenamesD[i], CV_LOAD_IMAGE_UNCHANGED);
            double tframe = vTimestamps[i];

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            sys.trackImage(imRGB, imDepth, tframe);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

            double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

            // Wait to load the next frame
            double T = 0;
            if (i < nImages - 1)
                T = vTimestamps[i + 1] - tframe;
            else if (i > 0)
                T = tframe - vTimestamps[i - 1];

            if (ttrack < T)
                usleep((T - ttrack) * 1e6);
        }
        else
        {
            i--;
        }
    }
}

void LoadImages(const std::string &datasetRootPath,
                std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD,
                std::vector<double> &vTimestamps)
{
    std::ifstream fAssociation;
    fAssociation.open((datasetRootPath + "association.txt").c_str());
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
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}