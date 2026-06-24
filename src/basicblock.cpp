#include "dlinf/blocks/basicblock.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dlinf {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

RowMatrixXf basicblock_direct(
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
    int stride,
    int padding,
    int height,
    int width,
    float bn_epsilon) {
    const auto spatial_size = static_cast<Eigen::Index>(height) * static_cast<Eigen::Index>(width);
    if (input.cols() != spatial_size) {
        throw std::invalid_argument("basicblock_direct input shape does not match height * width");
    }

    RowMatrixXf conv1 = dlinf::conv2d_naive_direct(
        input,
        conv1_weight,
        conv1_bias,
        stride,
        padding,
        height,
        width);

    RowMatrixXf bn1 = dlinf::batchnorm2d_direct(
        conv1,
        bn1_weight,
        bn1_bias,
        bn1_running_mean,
        bn1_running_var,
        bn_epsilon);


    relu_inplace(bn1.data(),  bn1.size()); // RowMatrixXf .data() returns float*, so that works
    RowMatrixXf conv2 = dlinf::conv2d_naive_direct(
        bn1, // Eigen array 
        conv2_weight, // TensorViewF32
        conv2_bias,
        1, // stride is always 1 in the second conv in a resnet
        padding,
        height,
        width);        

    RowMatrixXf bn2 = dlinf::batchnorm2d_direct(
        conv2,
        bn2_weight,
        bn2_bias,
        bn2_running_mean,
        bn2_running_var,
        bn_epsilon);
    if (bn2.cols() != spatial_size) {
        throw std::invalid_argument("basicblock_direct output shape does not match height * width");
    }

    // Identity skip connection: the residual tensor and the block output must
    // have the same [C, H, W] shape before they can be added.
    const std::vector<std::uint64_t> output_shape{
        static_cast<std::uint64_t>(bn2.rows()),
        static_cast<std::uint64_t>(height),
        static_cast<std::uint64_t>(width),
    };
    const std::vector<std::uint64_t> residual_shape{
        static_cast<std::uint64_t>(input.rows()),
        static_cast<std::uint64_t>(height),
        static_cast<std::uint64_t>(width),
    };

    // The skip connection: add all the output so far to the original input
    elementwise_add(bn2.data(), output_shape, input.data(), residual_shape);

    relu_inplace(bn2.data(), bn2.size());
    return bn2;
}
/*
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        */
}  // namespace dlinf
