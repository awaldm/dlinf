#pragma once

#include "dlinf/tensor.hpp"

#include <Eigen/Core>

namespace dlinf {

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> conv2d_naive_direct(
    const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias,
    int stride,
    int padding,
    int height,
    int width);

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> conv2d_im2col_eigen(
    const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias,
    int stride,
    int padding,
    int height,
    int width);

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> conv2d(
    const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias,
    int stride,
    int padding,
    int height,
    int width);

}  // namespace dlinf
