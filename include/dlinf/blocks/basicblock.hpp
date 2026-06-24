#pragma once

#include "dlinf/activation.hpp"
#include "dlinf/layers/batchnorm2d.hpp"
#include "dlinf/layers/conv2d.hpp"
#include "dlinf/ops/elementwise.hpp"
#include "dlinf/tensor.hpp"

#include <Eigen/Core>

namespace dlinf {

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> basicblock_direct(
    const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& input,
    const TensorViewF32& conv1_weight,
    const TensorViewF32& conv1_bias,
    const TensorViewF32& bn1_weight,
    const TensorViewF32& bn1_bias,
    const TensorViewF32& bn1_running_mean,
    const TensorViewF32& bn1_running_var,
    const TensorViewF32& conv2_weight,
    const TensorViewF32& conv2_bias,
    const TensorViewF32& bn2_weight,
    const TensorViewF32& bn2_bias,
    const TensorViewF32& bn2_running_mean,
    const TensorViewF32& bn2_running_var,
    const TensorViewF32& downsample_conv_weight,
    const TensorViewF32& downsample_conv_bias,
    const TensorViewF32& downsample_bn_weight,
    const TensorViewF32& downsample_bn_bias,
    const TensorViewF32& downsample_bn_running_mean,
    const TensorViewF32& downsample_bn_running_var,
    int stride,
    int padding,
    int height,
    int width,
    float bn_epsilon);

}  // namespace dlinf
