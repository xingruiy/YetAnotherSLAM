#pragma once
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include "utils/numType.h"

// Eigen's ostream operator is not compatible with ceres::Jet types.
// In particular, Eigen assumes that the scalar type (here Jet<T,N>) can be
// casted to an arithmetic type, which is not true for ceres::Jet.
// Unfortunately, the ceres::Jet class does not define a conversion
// operator (http://en.cppreference.com/w/cpp/language/cast_operator).
//
// This workaround creates a template specialization for Eigen's cast_impl,
// when casting from a ceres::Jet type. It relies on Eigen's internal API and
// might break with future versions of Eigen.
namespace Eigen
{
namespace internal
{

template <class T, int N, typename NewType>
struct cast_impl<ceres::Jet<T, N>, NewType>
{
    EIGEN_DEVICE_FUNC
    static inline NewType run(ceres::Jet<T, N> const &x)
    {
        return static_cast<NewType>(x.a);
    }
};

} // namespace internal
} // namespace Eigen

class LocalParameterizationSE3 : public ceres::LocalParameterization
{
public:
    virtual ~LocalParameterizationSE3() {}

    // SE3 plus operation for Ceres
    //
    //  T * exp(x)
    //
    virtual bool Plus(
        double const *T_raw,
        double const *delta_raw,
        double *T_plus_delta_raw) const
    {
        Eigen::Map<SE3 const> const T(T_raw);
        Eigen::Map<Vec6d const> const delta(delta_raw);
        Eigen::Map<SE3> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = T * SE3::exp(delta);
        return true;
    }

    // Jacobian of SE3 plus operation for Ceres
    //
    // Dx T * exp(x)  with  x=0
    //
    virtual bool ComputeJacobian(double const *T_raw, double *jacobian_raw) const
    {
        Eigen::Map<SE3 const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(jacobian_raw);
        jacobian = T.Dx_this_mul_exp_x_at_0();
        return true;
    }

    virtual int GlobalSize() const { return SE3::num_parameters; }

    virtual int LocalSize() const { return SE3::DoF; }
};

struct ReprojectionErrorFunctor
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ReprojectionErrorFunctor(double x, double y) : obsX(x), obsY(y) {}

    template <typename T>
    bool operator()(const T *K, const T *TData, const T *ptData, T *residual) const
    {
        const T &fx = K[0];
        const T &fy = K[1];
        const T &cx = K[2];
        const T &cy = K[3];

        Eigen::Map<Sophus::SE3<T> const> Twc(TData);
        Eigen::Map<Eigen::Matrix<T, 3, 1> const> pt(ptData);
        Eigen::Matrix<T, 3, 1> ptWarped = Twc * pt;

        T projX = fx * ptWarped(0) / ptWarped(2) + cx;
        T projY = fy * ptWarped(1) / ptWarped(2) + cy;

        residual[0] = projX - T(obsX);
        residual[1] = projY - T(obsY);

        return true;
    }

    static ceres::CostFunction *create(double x, double y)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionErrorFunctor, 2, 4, SE3::num_parameters, 3>(new ReprojectionErrorFunctor(x, y));
    }

    double obsX, obsY;
};

struct PointToPointErrorFunctor
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PointToPointErrorFunctor(double x, double y, double z) : obsX(x), obsY(y), obsZ(z) {}

    template <typename T>
    bool operator()(const T *TData, const T *ptData, T *residual) const
    {
        Eigen::Map<Sophus::SE3<T> const> Twc(TData);
        Eigen::Map<Eigen::Matrix<T, 3, 1> const> pt(ptData);
        Eigen::Matrix<T, 3, 1> ptTransformed = Twc * pt;

        residual[0] = ptTransformed(0) - T(obsX);
        residual[1] = ptTransformed(1) - T(obsY);
        residual[2] = ptTransformed(2) - T(obsZ);

        return true;
    }

    static ceres::CostFunction *create(double x, double y, double z)
    {
        return new ceres::AutoDiffCostFunction<PointToPointErrorFunctor, 3, SE3::num_parameters, 3>(new PointToPointErrorFunctor(x, y, z));
    }

    double obsX, obsY, obsZ;
};