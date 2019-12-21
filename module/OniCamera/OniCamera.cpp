#include "OniCamera.h"

namespace ONI
{

Camera::Camera() : Camera(640, 480, 30)
{
}

Camera::Camera(const int &w, const int &h, const int &fps)
    : mnWidth(w), mnHeight(h), mnFrameRate(fps)
{
    // openni context initialization
    if (OpenNI::initialize() != STATUS_OK)
    {
        std::cout << OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // openni camera open
    if (mDevice.open(ANY_DEVICE) != STATUS_OK)
    {
        std::cout << OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // create depth stream
    if (mDepthStream.create(mDevice, SENSOR_DEPTH) != STATUS_OK)
    {
        std::cout << OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // create colour stream
    if (mColourStream.create(mDevice, SENSOR_COLOR) != STATUS_OK)
    {
        std::cout << OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    auto videoMode = VideoMode();
    videoMode.setResolution(mnWidth, mnHeight);
    videoMode.setFps(mnFrameRate);
    videoMode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
    mDepthStream.setVideoMode(videoMode);

    videoMode.setPixelFormat(PIXEL_FORMAT_RGB888);
    mColourStream.setVideoMode(videoMode);

    if (mDevice.isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR))
        mDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

    mColourStream.setMirroringEnabled(false);
    mDepthStream.setMirroringEnabled(false);

    if (mDepthStream.start() != STATUS_OK)
    {
        std::cout << OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    if (mColourStream.start() != STATUS_OK)
    {
        std::cout << OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    std::cout << "Depth Camera Ready" << std::endl;
}

Camera::~Camera()
{
    mColourStream.stop();
    mColourStream.destroy();
    mDepthStream.stop();
    mDepthStream.destroy();
    mDevice.close();
    OpenNI::shutdown();

    std::cout << "Depth Camera Stopped" << std::endl;
    return;
}

bool Camera::TryFetchingImages(Mat &imDepth, Mat &imRGB)
{
    VideoStream *streams[] = {&mDepthStream, &mColourStream};

    int streamReady = -1;
    auto lastState = STATUS_OK;

    while (lastState == STATUS_OK)
    {
        lastState = OpenNI::waitForAnyStream(streams, 2, &streamReady, 0);

        if (lastState == STATUS_OK)
        {
            switch (streamReady)
            {
            case 0: //depth ready
                if (mDepthStream.readFrame(&mDepthFrameRef) == STATUS_OK)
                    imDepth = cv::Mat(mnHeight, mnWidth, CV_16UC1, const_cast<void *>(mDepthFrameRef.getData()));
                break;

            case 1: // color ready
                if (mColourStream.readFrame(&mColourFrameRef) == STATUS_OK)
                {
                    imRGB = cv::Mat(mnHeight, mnWidth, CV_8UC3, const_cast<void *>(mColourFrameRef.getData()));
                }
                break;

            default: // unexpected stream
                return false;
            }
        }
    }

    if (!mDepthFrameRef.isValid() || !mColourFrameRef.isValid())
        return false;

    return true;
}

} // namespace ONI