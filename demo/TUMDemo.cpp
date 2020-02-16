#include "System.h"
#include "TUM_Loader.h"

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

        imRGB = cv::imread(std::string(argv[3]) + "/" + vstrImageFilenamesRGB[i], CV_LOAD_IMAGE_UNCHANGED);
        imDepth = cv::imread(std::string(argv[3]) + "/" + vstrImageFilenamesD[i], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[i];
        sys.trackImage(imRGB, imDepth, tframe);
    }
}