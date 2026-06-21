#pragma once

#include "dlinf/tensor.hpp"
#include <Eigen/Core>

namespace dlinf {

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> batchnorm2d_direct(
    const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias,
    const TensorViewF32& running_mean,
    const TensorViewF32& running_var,
    float epsilon);

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> batchnorm2d(
    const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& input,
    const TensorViewF32& weight,         // gamma [C]
    const TensorViewF32& bias,           // beta  [C]
    const TensorViewF32& running_mean,   // [C]
    const TensorViewF32& running_var,    // [C]
    float epsilon);

}  // namespace dlinf
