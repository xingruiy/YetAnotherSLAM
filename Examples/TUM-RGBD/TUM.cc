#include "CoreSystem.h"
#include "utils/GlobalSettings.h"

void LoadImages(const std::string &path,
                std::vector<std::string> &imgFilename,
                std::vector<std::string> &depthFilename,
                std::vector<double> &timeStamps);

int main(int argc, char **argv)
{
    if (argc <= 3)
    {
        std::cout << "usage: ./TUM setting vocabulary imgPath" << std::endl;
        exit(-1);
    }

    std::vector<std::string> imgFilenames;
    std::vector<std::string> depthFilenames;
    std::vector<double> timeStamps;
    std::string filePath = std::string(argv[3]);
    LoadImages(filePath, imgFilenames, depthFilenames, timeStamps);

    cv::Mat imDepth, imRGB;
    size_t nImages = imgFilenames.size();
    if (imgFilenames.empty())
    {
        std::cerr << "No images found in provided path." << std::endl;
        return -1;
    }

    std::cout << "Images in the sequence: " << nImages << std::endl;

    slam::GlobalSettings settings(argv[1]);
    settings.printDebugInfo();
    slam::CoreSystem sys(&settings, argv[2]);

    for (int i = 0; i < nImages; ++i)
    {
        if (slam::g_bSystemKilled)
            break;

        if (slam::g_bSystemRunning)
        {
            imRGB = cv::imread(std::string(argv[3]) + "/" + imgFilenames[i], CV_LOAD_IMAGE_UNCHANGED);
            imDepth = cv::imread(std::string(argv[3]) + "/" + depthFilenames[i], CV_LOAD_IMAGE_UNCHANGED);
            double tframe = timeStamps[i];

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            sys.takeNewFrame(imRGB, imDepth, tframe);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

            double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

            // Wait to load the next frame
            double T = 0;
            if (i < nImages - 1)
                T = timeStamps[i + 1] - tframe;
            else if (i > 0)
                T = tframe - timeStamps[i - 1];

            if (ttrack < T)
                usleep((T - ttrack) * 1e6);
        }
        else
        {
            i--;
        }
    }

    sys.writeTrajectoryToFile("CameraTrajectory.txt");
}

void LoadImages(const std::string &path,
                std::vector<std::string> &imgFilename,
                std::vector<std::string> &depthFilename,
                std::vector<double> &timeStamps)
{
    std::ifstream fAssociation;
    fAssociation.open((path + "association.txt").c_str());
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
            timeStamps.push_back(t);
            ss >> sRGB;
            imgFilename.push_back(sRGB);
            ss >> t;
            ss >> sD;
            depthFilename.push_back(sD);
        }
    }
}