#include "DataStruct/frame.h"

Frame::Frame(Mat imRGB, Mat imDepth, Mat imGray, Mat nmap, Mat33d &K)
    : K(K), imRGB(imRGB.clone()), imGray(imGray.clone()),
      imDepth(imDepth.clone()), nmap(nmap.clone())
{
}
