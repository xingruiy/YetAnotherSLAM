#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

#include "DBoW2/DBoW2/FORB.h"
#include "DBoW2/DBoW2/TemplatedVocabulary.h"

namespace slam
{

static std::vector<cv::Mat> ToDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j = 0; j < Descriptors.rows; j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
    ORBVocabulary;

} // namespace slam

#endif // ORBVOCABULARY_H
