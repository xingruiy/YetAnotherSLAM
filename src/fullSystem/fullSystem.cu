#include "fullSystem/fullSystem.h"
#include "denseTracker/cudaImageProc.h"
#include <chrono>

FullSystem::FullSystem(const char *configFile)
{
}

FullSystem::FullSystem(int w, int h, Mat33d K, int numLvl, bool view)
    : currentState(-1)
{
    localMapper = std::make_shared<LocalMapper>(w, h, K);
    coarseTracker = std::make_shared<DenseTracker>(w, h, K, numLvl);

    lastTrackedPose = SE3(Mat44d::Identity());
    lastReferencePose = SE3(Mat44d::Identity());
}

void FullSystem::processFrame(Mat rawImage, Mat rawDepth)
{
    currentFrame = std::make_shared<Frame>(rawImage, rawDepth);
    cv::imwrite("Raw.png", rawImage);

    switch (currentState)
    {
    case -1:
    {
        coarseTracker->setReferenceFrame(currentFrame);
        fuseCurrentFrame();
        currentState = 0;
        break;
    }
    case 0:
    {
        auto rval = trackCurrentFrame();
        if (rval)
        {
            rawFramePoseHistory.push_back(lastTrackedPose);
            fuseCurrentFrame();
            if (needNewKF())
                createNewKF();
        }
        break;
    }
    }
}

bool FullSystem::trackCurrentFrame()
{
    coarseTracker->setTrackingFrame(currentFrame);
    SE3 rval = coarseTracker->getIncrementalTransform();
    lastTrackedPose = lastTrackedPose * rval.inverse();
    return true;
}

void FullSystem::fuseCurrentFrame()
{
    GMat currDepth = coarseTracker->getReferenceDepth();
    localMapper->fuseFrame(currDepth, lastTrackedPose);
    GMat vertex(480, 640, CV_32FC4);
    localMapper->raytrace(vertex, lastTrackedPose);
    std::cout << lastTrackedPose.matrix3x4() << std::endl;
    GMat nmap, scene;
    computeNormal(vertex, nmap);
    renderScene(vertex, nmap, scene);
    Mat map(scene);
    cv::imwrite("test.png", map);

    // cv::imshow("img", map);
    // cv::waitKey(1);
}

bool FullSystem::needNewKF()
{
    return false;
}

void FullSystem::createNewKF()
{
}

void FullSystem::resetSystem()
{
}

std::vector<SE3> FullSystem::getRawFramePoseHistory() const
{
    return rawFramePoseHistory;
}