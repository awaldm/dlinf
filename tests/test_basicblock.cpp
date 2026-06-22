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
        std::cerr << "usage: test_basicblock <resnet_weights.elw> <basicblock_golden.elw>\n";
        return 1;
    }

    const std::string archive_path = argv[1];
    const std::string golden_path = argv[2];

    try {
        const auto archive = dlinf::WeightArchive::load(archive_path);
        const auto golden = dlinf::WeightArchive::load(golden_path);

        // Get the network params
        const auto conv1_weight = archive.tensor_f32("layer1.0.conv1.weight");
        //const auto conv1_bias = archive.tensor_f32("layer1.0.conv1.bias");
        const auto bn1_weight = archive.tensor_f32("layer1.0.bn1.weight");
        const auto bn1_bias = archive.tensor_f32("layer1.0.bn1.bias");
        const auto bn1_running_mean = archive.tensor_f32("layer1.0.bn1.running_mean");
        const auto bn1_running_var = archive.tensor_f32("layer1.0.bn1.running_var");
        const auto conv2_weight = archive.tensor_f32("layer1.0.conv2.weight");
        //const auto conv2_bias = archive.tensor_f32("layer1.0.conv2.bias");
        const auto bn2_weight = archive.tensor_f32("layer1.0.bn2.weight");
        const auto bn2_bias = archive.tensor_f32("layer1.0.bn2.bias");
        const auto bn2_running_mean = archive.tensor_f32("layer1.0.bn2.running_mean");
        const auto bn2_running_var = archive.tensor_f32("layer1.0.bn2.running_var");

        // Get the basicblock input and the expected output (pytorch reference)
        const auto input_view = golden.tensor_f32("layer1.0.input");
        const auto expected_view = golden.tensor_f32("layer1.0.expected");

        const auto input = dlinf::test_support::map_chw_as_rows(input_view);
        const auto expected = dlinf::test_support::map_chw_as_rows(expected_view);

        // Get the derived block parameters
        const auto height = input_view.shape()[1];
        const auto width = input_view.shape()[2];
        const auto out_channels = conv1_weight.shape()[0];

        // And the constants
        const int stride = 1;
        const int padding = 1;
        const float bn_epsilon = 1e-5f;

        const auto zero_bias = dlinf::test_support::zero_bias_storage(out_channels);
        const auto conv1_bias = dlinf::test_support::bias_view(zero_bias);
        const auto conv2_bias = dlinf::test_support::bias_view(zero_bias);

        // Call the basicblock to run inference on the input
        const RowMatrixXf actual_direct = dlinf::basicblock_direct(
            input,
            conv1_weight,
            conv1_bias,
            bn1_weight,
            bn1_bias,
            bn1_running_mean,
            bn1_running_var,
            conv2_weight,
            conv2_bias,
            bn2_weight,
            bn2_bias,
            bn2_running_mean,
            bn2_running_var,
            stride,
            padding,
            height,
            width,
            bn_epsilon);

        std::cout << "BasicBlock golden test\n";
        std::cout << "  weights archive: " << archive_path << "\n";
        std::cout << "  golden archive:  " << golden_path << "\n";
        dlinf::test_support::print_tensor_shape("input shape:    ", input_view.shape());
        dlinf::test_support::print_tensor_shape("expected tensor shape: ", expected_view.shape());
        std::cout << "  expected comparison view: [" << expected.rows() << " x " << expected.cols() << "]\n";
        std::cout << "  actual comparison view:   [" << actual_direct.rows() << " x " << actual_direct.cols() << "]\n";

        if (actual_direct.rows() != expected.rows() ||
            actual_direct.cols() != expected.cols()) {
            std::cout << "  shape check:    actual/expected shape mismatch\n";
            std::cout << "FAIL\n";
            return 1;
        }

        const float direct_max_abs_error = dlinf::test_support::max_abs_error(actual_direct, expected);
        std::cout << "  direct max abs error:  " << direct_max_abs_error << "\n";
        std::cout << "  tolerance:      " << kTolerance << "\n";

        if (direct_max_abs_error > kTolerance) {
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
