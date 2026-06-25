#pragma once

#include "dlinf/weight_archive.hpp"

#include <Eigen/Core>

namespace dlinf {

Eigen::VectorXf resnet18_direct(
    const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& input,
    const WeightArchive& weights,
    int input_height,
    int input_width);

}  // namespace dlinf
