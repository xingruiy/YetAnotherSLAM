#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

void LoadImages(const std::string &DatasetRootPath,
                std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD,
                std::vector<double> &vTimestamps);
