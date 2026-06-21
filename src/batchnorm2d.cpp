#include "dlinf/layers/batchnorm2d.hpp"

#include <cmath>
#include <stdexcept>

namespace dlinf {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

RowMatrixXf batchnorm2d_direct(
    const Eigen::Ref<const RowMatrixXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias,
    const TensorViewF32& running_mean,
    const TensorViewF32& running_var,
    const float epsilon) {
    if (weight.shape().size() != 1) {
        throw std::runtime_error("BatchNorm2D weight must be rank 1");
    }
    if (bias.shape().size() != 1) {
        throw std::runtime_error("BatchNorm2D bias must be rank 1");
    }
    if (running_mean.shape().size() != 1) {
        throw std::runtime_error("BatchNorm2D running_mean must be rank 1");
    }
    if (running_var.shape().size() != 1) {
        throw std::runtime_error("BatchNorm2D running_var must be rank 1");
    }

    const auto C_out = static_cast<Eigen::Index>(weight.shape()[0]);

    if (input.rows() != C_out) {
        throw std::runtime_error("BatchNorm2D input channel mismatch");
    }
    if (static_cast<Eigen::Index>(bias.size()) != C_out) {
        throw std::runtime_error("BatchNorm2D bias size mismatch");
    }
    if (static_cast<Eigen::Index>(running_mean.size()) != C_out) {
        throw std::runtime_error("BatchNorm2D running_mean size mismatch");
    }
    if (static_cast<Eigen::Index>(running_var.size()) != C_out) {
        throw std::runtime_error("BatchNorm2D running_var size mismatch");
    }

    RowMatrixXf output(C_out, input.cols());

    for (Eigen::Index oc = 0; oc < C_out; ++oc) {
        const float inv_std = 1.0f / std::sqrt(running_var.data()[oc] + epsilon);
        for (Eigen::Index element = 0; element < input.cols(); ++element) {
            output(oc, element) = weight.data()[oc] * (input(oc, element) - running_mean.data()[oc]) * inv_std + bias.data()[oc];
        }
    }

    return output;
}

RowMatrixXf batchnorm2d(
    const Eigen::Ref<const RowMatrixXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias,
    const TensorViewF32& running_mean,
    const TensorViewF32& running_var,
    float epsilon) {
    return batchnorm2d_direct(input, weight, bias, running_mean, running_var, epsilon);
}

}  // namespace dlinf
