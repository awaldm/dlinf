// This lives in support/ instead of include/ because it's part of the test harness, not of the library
#pragma once

#include "dlinf/tensor.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace dlinf::test_support {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline void print_shape(const std::vector<std::uint64_t>& shape) {
    std::cout << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << shape[i];
    }
    std::cout << "]";
}

inline void print_tensor_shape(const char* label, const std::vector<std::uint64_t>& shape) {
    std::cout << "  " << label;
    print_shape(shape);
    std::cout << "\n";
}

inline Eigen::Map<const RowMatrixXf> map_chw_as_rows(const TensorViewF32& view) {
    const auto& shape = view.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("map_chw_as_rows expects a rank-3 [C, H, W] tensor");
    }
    return Eigen::Map<const RowMatrixXf>(
        view.data(),
        static_cast<Eigen::Index>(shape[0]),
        static_cast<Eigen::Index>(shape[1] * shape[2]));
}

inline Eigen::Map<const Eigen::VectorXf> map_vector(const TensorViewF32& view) {
    if (view.shape().size() != 1) {
        throw std::invalid_argument("map_vector expects a rank-1 tensor");
    }
    return Eigen::Map<const Eigen::VectorXf>(
        view.data(),
        static_cast<Eigen::Index>(view.size()));
}

inline float max_abs_error(const RowMatrixXf& actual, const RowMatrixXf& expected) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        throw std::invalid_argument("matrix max_abs_error requires matching shapes");
    }
    return (actual - expected).cwiseAbs().maxCoeff();
}

inline float max_abs_error(const Eigen::VectorXf& actual, const Eigen::VectorXf& expected) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("vector max_abs_error requires matching sizes");
    }
    return (actual - expected).cwiseAbs().maxCoeff();
}

inline std::vector<float> zero_bias_storage(std::uint64_t channels) {
    return std::vector<float>(static_cast<std::size_t>(channels), 0.0f);
}

inline TensorViewF32 bias_view(const std::vector<float>& storage) {
    return TensorViewF32(storage.data(), {static_cast<std::uint64_t>(storage.size())});
}

}  // namespace dlinf::test_support
