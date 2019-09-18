#include "fullSystem/fullSystem.h"
#include "denseTracker/cudaImageProc.h"

FullSystem::FullSystem(const char *configFile)
{
}

FullSystem::FullSystem(int w, int h, Mat33d K, int numLvl, bool view)
    : currentState(-1)
{
    // localMapper = std::make_shared<LocalMapper>(w, h, K);
    localMapper = std::make_shared<DenseMapping>(w, h, K);
    coarseTracker = std::make_shared<DenseTracker>(w, h, K, numLvl);

    lastTrackedPose = SE3(Mat44d::Identity());
    lastReferencePose = SE3(Mat44d::Identity());
}

void FullSystem::processFrame(Mat rawImage, Mat rawDepth)
{
    currentFrame = std::make_shared<Frame>(rawImage, rawDepth);

    switch (currentState)
    {
    case -1:
    {
        coarseTracker->setReferenceFrame(currentFrame);
        fuseCurrentFrame(lastTrackedPose);
        currentState = 0;
        break;
    }
    case 0:
    {
        auto rval = trackCurrentFrame();
        if (rval)
        {
            fuseCurrentFrame(lastTrackedPose);
            updateLocalMapObservation(lastTrackedPose);
            rawFramePoseHistory.push_back(lastTrackedPose);

            if (needNewKF())
                createNewKF();
        }
        else
        {
            currentState = 1;
        }

        break;
    }
    case 1:
        printf("tracking loast, attempt to resuming...\n");
        break;
    }
}

bool FullSystem::trackCurrentFrame()
{
    coarseTracker->setTrackingFrame(currentFrame);
    SE3 rval = coarseTracker->getIncrementalTransform();
    lastTrackedPose = lastTrackedPose * rval.inverse();
    return true;
}

void FullSystem::fuseCurrentFrame(const SE3 &T)
{
    GMat currDepth = coarseTracker->getReferenceDepth();
    // Mat depth(currDepth);
    // cv::imshow("depth", depth);
    // cv::waitKey(1);
    localMapper->fuseFrame(currDepth, T);
}

void FullSystem::updateLocalMapObservation(const SE3 &T)
{
    GMat vertex(480, 640, CV_32FC4);
    localMapper->raytrace(vertex, T);

    GMat nmap, scene;
    computeNormal(vertex, nmap);
    renderScene(vertex, nmap, scene);
    Mat map(scene);

    cv::imshow("img", map);
    cv::waitKey(1);
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
    localMapper->reset();
}

std::vector<SE3> FullSystem::getRawFramePoseHistory() const
{
    return rawFramePoseHistory;
}

size_t FullSystem::getMesh(float *vbuffer, float *nbuffer, size_t bufferSize)
{
    return localMapper->fetch_mesh_with_normal(vbuffer, nbuffer);
}