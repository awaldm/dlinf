/*
ArrayXf is a 1-dim Eigen array


*/

#pragma once

#include <Eigen/Core>

#include <cstddef>

namespace dlinf {

inline void relu_inplace(float* data, std::size_t size) {
    Eigen::Map<Eigen::ArrayXf> values(data, static_cast<Eigen::Index>(size)); // map data to ArrayXf, a zero-copy view. Eigen::ArrayXf::Map values would also work (alias declaration)
    values = values.max(0.0f);
}

}  // namespace dlinf

