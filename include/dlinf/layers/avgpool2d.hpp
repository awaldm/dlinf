#pragma once

#include <Eigen/Core>

namespace dlinf {

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> avgpool2d_direct(
    const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& input,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding);

}  // namespace dlinf
