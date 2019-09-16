#pragma once
#include <iostream>
#include <fstream>
#include "NumType.h"

void TUMLoad(
    const std::string &baseDir,
    std::vector<double> &listOfTimeStamp,
    std::vector<std::string> &listOfDepth,
    std::vector<std::string> &listOfImage)
{
    std::ifstream file(baseDir + "association.txt");

    if (file.is_open())
    {
        double ts1, ts2;
        std::string pDepth;
        std::string pImage;

        while (file >> ts1 >> pImage >> ts2 >> pDepth)
        {
            listOfTimeStamp.push_back(ts1);
            listOfDepth.push_back(baseDir + pDepth);
            listOfImage.push_back(baseDir + pImage);
        }
    }
}

void TUMSave(
    const std::string &path,
    std::vector<double> listTimeStamp,
    std::vector<SE3d> trajectory)
{
    std::ofstream file(path + "result.txt", std::ios_base::out);

    if (file.is_open())
    {
        for (int i = 0; i < trajectory.size(); ++i)
        {
            if (i >= listTimeStamp.size())
                break;

            double ts = listTimeStamp[i];
            const auto &pose = trajectory[i];
            Eigen::Vector3d t = pose.translation();
            Eigen::Quaterniond q(pose.rotationMatrix());

            file << std::fixed
                 << std::setprecision(4)
                 << ts << " "
                 << t(0) << " "
                 << t(1) << " "
                 << t(2) << " "
                 << q.x() << " "
                 << q.y() << " "
                 << q.z() << " "
                 << q.w() << std::endl;
        }

        file.close();
    }
}