#include "dlinf/blocks/basicblock.hpp"
#include "dlinf/weight_archive.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr float kTolerance = 2e-5f;

void print_shape(const std::vector<std::uint64_t>& shape) {
    std::cout << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << shape[i];
    }
    std::cout << "]";
}

void print_tensor_shape(const char* label, const std::vector<std::uint64_t>& shape) {
    std::cout << "  " << label;
    print_shape(shape);
    std::cout << "\n";
}

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

        // Cast both to RowMatrixXf
        Eigen::Map<const RowMatrixXf> input(input_view.data(), input_view.shape()[0],
                                             input_view.shape()[1] * input_view.shape()[2]);
        Eigen::Map<const RowMatrixXf> expected(expected_view.data(), expected_view.shape()[0],
                                                expected_view.shape()[1] * expected_view.shape()[2]);

        // Get the derived block parameters
        const auto height = input_view.shape()[1];
        const auto width = input_view.shape()[2];
        const auto out_channels = conv1_weight.shape()[0];

        // And the constants
        const int stride = 1;
        const int padding = 1;
        const float bn_epsilon = 1e-5f;

        std::vector<float> zero_bias(out_channels, 0.0f);
        dlinf::TensorViewF32 conv1_bias(zero_bias.data(), {out_channels}); // shape must be a vector of uint64_t, therefore the {}        
        dlinf::TensorViewF32 conv2_bias(zero_bias.data(), {out_channels}); // shape must be a vector of uint64_t, therefore the {}        

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
        print_tensor_shape("input shape:    ", input_view.shape());
        print_tensor_shape("expected tensor shape: ", expected_view.shape());
        std::cout << "  expected comparison view: [" << expected.rows() << " x " << expected.cols() << "]\n";
        std::cout << "  actual comparison view:   [" << actual_direct.rows() << " x " << actual_direct.cols() << "]\n";

        if (actual_direct.rows() != expected.rows() ||
            actual_direct.cols() != expected.cols()) {
            std::cout << "  shape check:    actual/expected shape mismatch\n";
            std::cout << "FAIL\n";
            return 1;
        }

        const float direct_max_abs_error = (actual_direct - expected).cwiseAbs().maxCoeff();
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
