#include <ceres/ceres.h>
#include "utils/numType.h"
#include "utils/costFunctors.h"

bool test(SE3 &a, SE3 &b, Vec3d &pt, const Vec2d &proj0, const Vec2d &proj1, Vec4d &K)
{
    ceres::Problem problem;
    problem.AddParameterBlock(a.data(), SE3::num_parameters, new LocalParameterizationSE3);
    problem.SetParameterBlockConstant(a.data());
    problem.AddParameterBlock(b.data(), SE3::num_parameters, new LocalParameterizationSE3);
    // problem.SetParameterBlockConstant(b.data());

    problem.AddResidualBlock(
        ReprojectionErrorFunctor::create(proj0(0), proj0(1)),
        NULL,
        K.data(),
        a.data(),
        pt.data());
    problem.AddResidualBlock(
        ReprojectionErrorFunctor::create(proj1(0), proj1(1)),
        NULL,
        K.data(),
        b.data(),
        pt.data());
    problem.SetParameterBlockConstant(K.data());

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
}

int main(int argc, char **argv)
{
    Vec4d K = {580, 580, 320, 240};
    SE3 a = SE3();
    Vec2d proj0 = {200.0, 140.0};
    Vec3d pt = {(200.0 - 320.0) / 580.0 * 3.0, (140.0 - 240.0) / 580.0 * 3.0, 3.0};
    SE3 b = SE3(SO3::exp(SE3::Point(0.2, 0.5, 0.0)), SE3::Point(0, 0, 0)) *
            SE3(SO3::exp(SE3::Point(0.1, 0, 0)), SE3::Point(0, 0, 0)) *
            SE3(SO3::exp(SE3::Point(-0.2, -0.5, -0.0)), SE3::Point(0, 0, 0));
    Vec3d ptTransformed = b.inverse() * pt;
    Vec2d proj1;
    proj1(0) = 580.0 * ptTransformed(0) / ptTransformed(2) + 320.0;
    proj1(1) = 580.0 * ptTransformed(1) / ptTransformed(2) + 240.0;
    std::cout << b.matrix() << std::endl;
    std::cout << pt << std::endl;
    std::cout << proj1 << std::endl;

    test(a, b, pt, proj0, Vec2d(200, 190), K);
    std::cout << b.matrix() << std::endl;
    std::cout << pt << std::endl;
    ptTransformed = b.inverse() * pt;
    proj1(0) = 580.0 * ptTransformed(0) / ptTransformed(2) + 320.0;
    proj1(1) = 580.0 * ptTransformed(1) / ptTransformed(2) + 240.0;
    std::cout << proj1 << std::endl;
}