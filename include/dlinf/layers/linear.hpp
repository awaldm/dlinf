#pragma once

#include "dlinf/tensor.hpp"

#include <Eigen/Core>

namespace dlinf {

Eigen::VectorXf linear_naive(
    const Eigen::Ref<const Eigen::VectorXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias);

Eigen::VectorXf linear_eigen(
    const Eigen::Ref<const Eigen::VectorXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias);

Eigen::VectorXf linear(
    const Eigen::Ref<const Eigen::VectorXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias);

}  // namespace dlinf
