#include "oniCamera.h"

ONICamera::ONICamera() : ONICamera(640, 480, 30)
{
}

ONICamera::ONICamera(int w, int h, int fps) : width(w), height(h), frameRate(fps)
{
    // openni context initialization
    if (openni::OpenNI::initialize() != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // openni camera open
    if (device.open(openni::ANY_DEVICE) != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // create depth stream
    if (depthStream.create(device, openni::SENSOR_DEPTH) != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // create colour stream
    if (colourStream.create(device, openni::SENSOR_COLOR) != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    auto videoMode = openni::VideoMode();
    videoMode.setResolution(width, height);
    videoMode.setFps(frameRate);
    videoMode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
    depthStream.setVideoMode(videoMode);

    videoMode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
    colourStream.setVideoMode(videoMode);

    if (device.isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR))
        device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

    colourStream.setMirroringEnabled(false);
    depthStream.setMirroringEnabled(false);

    if (depthStream.start() != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    if (colourStream.start() != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    std::cout << "Depth Camera Ready" << std::endl;
}

ONICamera::~ONICamera()
{
    colourStream.stop();
    colourStream.destroy();
    depthStream.stop();
    depthStream.destroy();
    device.close();
    openni::OpenNI::shutdown();

    std::cout << "Depth Camera Stopped" << std::endl;
    return;
}

bool ONICamera::getNextImages(cv::Mat &depth, cv::Mat &image)
{
    openni::VideoStream *streams[] = {&depthStream, &colourStream};

    int streamReady = -1;
    auto lastState = openni::STATUS_OK;

    while (lastState == openni::STATUS_OK)
    {
        lastState = openni::OpenNI::waitForAnyStream(streams, 2, &streamReady, 0);

        if (lastState == openni::STATUS_OK)
        {
            switch (streamReady)
            {
            case 0: //depth ready
                if (depthStream.readFrame(&depthFrameRef) == openni::STATUS_OK)
                    depth = cv::Mat(height, width, CV_16UC1, const_cast<void *>(depthFrameRef.getData()));
                break;

            case 1: // color ready
                if (colourStream.readFrame(&colourFrameRef) == openni::STATUS_OK)
                {
                    image = cv::Mat(height, width, CV_8UC3, const_cast<void *>(colourFrameRef.getData()));
                }
                break;

            default: // unexpected stream
                return false;
            }
        }
    }

    if (!depthFrameRef.isValid() || !colourFrameRef.isValid())
        return false;

    return true;
}