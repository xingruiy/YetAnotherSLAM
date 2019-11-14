#include "dataStruct/frame.h"

Frame::Frame(Mat imRGB, Mat imDepth, Mat imGray, Mat nmap, Mat33d &K)
    : K(K)
{
    nmap.copyTo(this->nmap);
    imRGB.copyTo(this->imRGB);
    imGray.copyTo(this->imGray);
    imDepth.copyTo(this->imDepth);
}
