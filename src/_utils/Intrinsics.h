#ifndef _INTRINSICS_H
#define _INTRINSICS_H

struct Intrinsics
{
    inline Intrinsics(int w, int h, float fx, float fy, float cx, float cy)
        : fx(fx), fy(fy), cx(cx), cy(cy), w(w), h(h)
    {
        ifx = 1.f / fx;
        ify = i.f / fy;
    }

    inline Intrinsics pyrDown() const
    {
        return Intrinsics(0.5f * w, 0.5f * h, 0.5f * fx, 0.5f * fy, 0.5f * cx, 0.5f * cy);
    }

    int w, h;
    float fx, fy, cx, cy;
    float ifx, ify;
};

#endif