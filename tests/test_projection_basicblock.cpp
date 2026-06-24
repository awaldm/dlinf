#include "dlinf/blocks/basicblock.hpp"
#include "dlinf/weight_archive.hpp"
#include "test_support.hpp"

#include <Eigen/Core>

#include <exception>
#include <iostream>
#include <string>

namespace {

using RowMatrixXf = dlinf::test_support::RowMatrixXf;

constexpr float kTolerance = 2e-5f;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_projection_basicblock <resnet_weights.elw> <projection_basicblock_golden.elw>\n";
        return 1;
    }

    const std::string archive_path = argv[1];
    const std::string golden_path = argv[2];

    try {
        const auto archive = dlinf::WeightArchive::load(archive_path);
        const auto golden = dlinf::WeightArchive::load(golden_path);

        // Main path weights
        const auto conv1_weight = archive.tensor_f32("layer2.0.conv1.weight");
        const auto bn1_weight = archive.tensor_f32("layer2.0.bn1.weight");
        const auto bn1_bias = archive.tensor_f32("layer2.0.bn1.bias");
        const auto bn1_running_mean = archive.tensor_f32("layer2.0.bn1.running_mean");
        const auto bn1_running_var = archive.tensor_f32("layer2.0.bn1.running_var");
        const auto conv2_weight = archive.tensor_f32("layer2.0.conv2.weight");
        const auto bn2_weight = archive.tensor_f32("layer2.0.bn2.weight");
        const auto bn2_bias = archive.tensor_f32("layer2.0.bn2.bias");
        const auto bn2_running_mean = archive.tensor_f32("layer2.0.bn2.running_mean");
        const auto bn2_running_var = archive.tensor_f32("layer2.0.bn2.running_var");

        // Downsample (skip path) weights: 1x1 conv + BN
        const auto ds_conv_weight = archive.tensor_f32("layer2.0.downsample.0.weight");
        const auto ds_bn_weight = archive.tensor_f32("layer2.0.downsample.1.weight");
        const auto ds_bn_bias = archive.tensor_f32("layer2.0.downsample.1.bias");
        const auto ds_bn_running_mean = archive.tensor_f32("layer2.0.downsample.1.running_mean");
        const auto ds_bn_running_var = archive.tensor_f32("layer2.0.downsample.1.running_var");

        // Zero biases for conv1/conv2 (ResNet BasicBlock convs have bias=False)
        const auto out_channels = conv1_weight.shape()[0];
        const auto zero_bias = dlinf::test_support::zero_bias_storage(out_channels);
        const auto conv1_bias = dlinf::test_support::bias_view(zero_bias);
        const auto conv2_bias = dlinf::test_support::bias_view(zero_bias);

        // Downsample conv also has bias=False
        const auto ds_out_channels = ds_conv_weight.shape()[0];
        const auto ds_zero_bias = dlinf::test_support::zero_bias_storage(ds_out_channels);
        const auto ds_conv_bias = dlinf::test_support::bias_view(ds_zero_bias);

        const auto input_view = golden.tensor_f32("layer2.0.input");
        const auto expected_view = golden.tensor_f32("layer2.0.expected");

        const auto input = dlinf::test_support::map_chw_as_rows(input_view);
        const auto expected = dlinf::test_support::map_chw_as_rows(expected_view);

        const auto height = input_view.shape()[1];
        const auto width = input_view.shape()[2];
        const int stride = 2;
        const int padding = 1;
        const float bn_epsilon = 1e-5f;

        const RowMatrixXf actual = dlinf::basicblock_direct(
            input,
            conv1_weight, conv1_bias,
            bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
            conv2_weight, conv2_bias,
            bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
            ds_conv_weight, ds_conv_bias,
            ds_bn_weight, ds_bn_bias, ds_bn_running_mean, ds_bn_running_var,
            stride, padding, height, width, bn_epsilon);

        std::cout << "Projection BasicBlock golden test\n";
        std::cout << "  weights archive: " << archive_path << "\n";
        std::cout << "  golden archive:  " << golden_path << "\n";
        dlinf::test_support::print_tensor_shape("input shape:    ", input_view.shape());
        dlinf::test_support::print_tensor_shape("expected tensor shape: ", expected_view.shape());
        std::cout << "  expected comparison view: [" << expected.rows() << " x " << expected.cols() << "]\n";
        std::cout << "  actual comparison view:   [" << actual.rows() << " x " << actual.cols() << "]\n";

        if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
            std::cout << "  shape check:    actual/expected shape mismatch\n";
            std::cout << "FAIL\n";
            return 1;
        }

        const float max_abs_error = dlinf::test_support::max_abs_error(actual, expected);
        std::cout << "  max abs error:  " << max_abs_error << "\n";
        std::cout << "  tolerance:      " << kTolerance << "\n";

        if (max_abs_error > kTolerance) {
            std::cout << "FAIL\n";
            return 1;
        }

        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "FAIL: " << exc.what() << "\n";
        return 1;
    }
}
