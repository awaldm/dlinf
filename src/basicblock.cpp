#include "dlinf/blocks/basicblock.hpp"
#include "dlinf/activation.hpp"
#include "dlinf/layers/batchnorm2d.hpp"
#include "dlinf/layers/conv2d.hpp"
#include "dlinf/ops/elementwise.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dlinf {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

RowMatrixXf basicblock_direct(
    const Eigen::Ref<const RowMatrixXf>& input,
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
    float bn_epsilon) {

    const auto spatial_size = static_cast<Eigen::Index>(height) * static_cast<Eigen::Index>(width);
    if (input.cols() != spatial_size) {
        throw std::invalid_argument("basicblock_direct input shape does not match height * width");
    }

    // conv1
    auto out = conv2d_naive_direct(input, conv1_weight, conv1_bias, stride, padding, height, width);
    auto H_mid = (height + 2 * padding - 3) / stride + 1;
    auto W_mid = (width + 2 * padding - 3) / stride + 1;

    // bn1
    out = batchnorm2d_direct(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn_epsilon);

    // relu
    relu_inplace(out.data(), out.size());

    // conv2 (always stride=1)
    out = conv2d_naive_direct(out, conv2_weight, conv2_bias, 1, padding, H_mid, W_mid);
    auto H_out = (H_mid + 2 * padding - 3) / 1 + 1;
    auto W_out = (W_mid + 2 * padding - 3) / 1 + 1;

    // bn2
    out = batchnorm2d_direct(out, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, bn_epsilon);

    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("basicblock_direct output shape is empty");
    }

    // skip connection: projection if dimensions differ, identity otherwise
    const auto out_spatial = static_cast<Eigen::Index>(H_out) * static_cast<Eigen::Index>(W_out);
    if (out.rows() != input.rows() || out_spatial != spatial_size) {
        // Projection: 1x1 conv + BN on skip input to match channel and spatial dims
        auto skip = conv2d_naive_direct(input, downsample_conv_weight, downsample_conv_bias,
                                         stride, 0, height, width);
        skip = batchnorm2d_direct(skip, downsample_bn_weight, downsample_bn_bias,
                                   downsample_bn_running_mean, downsample_bn_running_var, bn_epsilon);

        const std::vector<std::uint64_t> out_shape{
            static_cast<std::uint64_t>(out.rows()),
            static_cast<std::uint64_t>(H_out),
            static_cast<std::uint64_t>(W_out),
        };
        const std::vector<std::uint64_t> skip_shape{
            static_cast<std::uint64_t>(skip.rows()),
            static_cast<std::uint64_t>(H_out),
            static_cast<std::uint64_t>(W_out),
        };
        elementwise_add(out.data(), out_shape, skip.data(), skip_shape);
    } else {
        const std::vector<std::uint64_t> out_shape{
            static_cast<std::uint64_t>(out.rows()),
            static_cast<std::uint64_t>(H_out),
            static_cast<std::uint64_t>(W_out),
        };
        const std::vector<std::uint64_t> input_shape{
            static_cast<std::uint64_t>(input.rows()),
            static_cast<std::uint64_t>(height),
            static_cast<std::uint64_t>(width),
        };
        elementwise_add(out.data(), out_shape, input.data(), input_shape);
    }

    // final relu
    relu_inplace(out.data(), out.size());
    return out;
}

}  // namespace dlinf
