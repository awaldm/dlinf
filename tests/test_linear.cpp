#include "dlinf/layers/linear.hpp"
#include "dlinf/weight_archive.hpp"
#include "test_support.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace {

constexpr float kTolerance = 1e-5f;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_linear <resnet_weights.elw> <fc_golden.elw>\n";
        return 1;
    }

    const std::string archive_path = argv[1];
    const std::string golden_path = argv[2];

    try {
        const auto archive = dlinf::WeightArchive::load(archive_path);
        const auto golden = dlinf::WeightArchive::load(golden_path);

        const auto fc_weight = archive.tensor_f32("fc.weight");
        const auto fc_bias = archive.tensor_f32("fc.bias");
        const auto input_view = golden.tensor_f32("fc.input");
        const auto expected_view = golden.tensor_f32("fc.expected");

        const auto input = dlinf::test_support::map_vector(input_view);
        const auto expected = dlinf::test_support::map_vector(expected_view);

        const Eigen::VectorXf actual_naive = dlinf::linear_naive(input, fc_weight, fc_bias);
        const Eigen::VectorXf actual_eigen = dlinf::linear_eigen(input, fc_weight, fc_bias);
        const Eigen::VectorXf actual_default = dlinf::linear(input, fc_weight, fc_bias);

        std::cout << "Linear golden test\n";
        std::cout << "  weights archive: " << archive_path << "\n";
        std::cout << "  golden archive:  " << golden_path << "\n";
        dlinf::test_support::print_tensor_shape("input shape:    ", input_view.shape());
        dlinf::test_support::print_tensor_shape("weight shape:   ", fc_weight.shape());
        dlinf::test_support::print_tensor_shape("bias shape:     ", fc_bias.shape());
        dlinf::test_support::print_tensor_shape("expected shape: ", expected_view.shape());
        std::cout << "  actual shape:   [" << actual_eigen.size() << "]\n";

        if (actual_naive.size() != expected.size() ||
            actual_eigen.size() != expected.size() ||
            actual_default.size() != expected.size()) {
            std::cout << "  shape check:    actual/expected size mismatch\n";
            std::cout << "FAIL\n";
            return 1;
        }

        const float naive_max_abs_error = dlinf::test_support::max_abs_error(actual_naive, expected);
        const float eigen_max_abs_error = dlinf::test_support::max_abs_error(actual_eigen, expected);
        const float default_max_abs_error = dlinf::test_support::max_abs_error(actual_default, expected);
        const float variant_max_abs_delta = dlinf::test_support::max_abs_error(actual_naive, actual_eigen);
        std::cout << "  naive max abs error:   " << naive_max_abs_error << "\n";
        std::cout << "  eigen max abs error:   " << eigen_max_abs_error << "\n";
        std::cout << "  default max abs error: " << default_max_abs_error << "\n";
        std::cout << "  naive/eigen max delta: " << variant_max_abs_delta << "\n";
        std::cout << "  tolerance:      " << kTolerance << "\n";

        if (naive_max_abs_error > kTolerance ||
            eigen_max_abs_error > kTolerance ||
            default_max_abs_error > kTolerance ||
            variant_max_abs_delta > kTolerance) {
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
