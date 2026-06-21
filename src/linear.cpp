#include "dlinf/layers/linear.hpp"

#include <stdexcept>

namespace eigen_learn {

namespace {

void validate_linear_inputs(
    const Eigen::Ref<const Eigen::VectorXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias) {
    if (weight.shape().size() != 2) {
        throw std::runtime_error("Linear weight must be rank 2");
    }
    if (bias.shape().size() != 1) {
        throw std::runtime_error("Linear bias must be rank 1");
    }
    // Input and output features/dimensions
    const auto out_features = static_cast<Eigen::Index>(weight.shape()[0]);
    const auto in_features = static_cast<Eigen::Index>(weight.shape()[1]);
    // do the checks on the sizes, they need to match
    if (input.size() != in_features) {
        throw std::runtime_error("Linear input size mismatch");
    }
    if (static_cast<Eigen::Index>(bias.size()) != out_features) {
        throw std::runtime_error("Linear bias size mismatch");
    }
}

}  // namespace

Eigen::VectorXf linear_naive(
    const Eigen::Ref<const Eigen::VectorXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias) {
    validate_linear_inputs(input, weight, bias);

    const auto out_features = static_cast<Eigen::Index>(weight.shape()[0]);
    const auto in_features = static_cast<Eigen::Index>(weight.shape()[1]);

    Eigen::VectorXf output(out_features);
    for (Eigen::Index out = 0; out < out_features; ++out) {
        float sum = bias.data()[out];
        for (Eigen::Index in = 0; in < in_features; ++in) {
            sum += weight.data()[out * in_features + in] * input[in];
        }
        output[out] = sum;
    }

    return output;
}

Eigen::VectorXf linear_eigen(
    const Eigen::Ref<const Eigen::VectorXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias) {
    validate_linear_inputs(input, weight, bias);

    const auto out_features = static_cast<Eigen::Index>(weight.shape()[0]);
    const auto in_features = static_cast<Eigen::Index>(weight.shape()[1]);

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        w(weight.data(), out_features, in_features);
    Eigen::Map<const Eigen::VectorXf> b(bias.data(), out_features);

    return w * input + b;
}

Eigen::VectorXf linear(
    const Eigen::Ref<const Eigen::VectorXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias) {
    return linear_eigen(input, weight, bias);
}

}  // namespace eigen_learn
