#include "dlinf/layers/linear.hpp"
#include "dlinf/weight_archive.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kTolerance = 1e-5f;

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
        std::cerr << "usage: test_linear <resnet_weights.elw> <fc_golden.elw>\n";
        return 1;
    }

    const std::string archive_path = argv[1];
    const std::string golden_path = argv[2];

    try {
        const auto archive = eigen_learn::WeightArchive::load(archive_path);
        const auto golden = eigen_learn::WeightArchive::load(golden_path);

        const auto fc_weight = archive.tensor_f32("fc.weight");
        const auto fc_bias = archive.tensor_f32("fc.bias");
        const auto input_view = golden.tensor_f32("fc.input");
        const auto expected_view = golden.tensor_f32("fc.expected");

        Eigen::Map<const Eigen::VectorXf> input(input_view.data(), input_view.size());
        Eigen::Map<const Eigen::VectorXf> expected(expected_view.data(), expected_view.size());

        const Eigen::VectorXf actual_naive = eigen_learn::linear_naive(input, fc_weight, fc_bias);
        const Eigen::VectorXf actual_eigen = eigen_learn::linear_eigen(input, fc_weight, fc_bias);
        const Eigen::VectorXf actual_default = eigen_learn::linear(input, fc_weight, fc_bias);

        std::cout << "Linear golden test\n";
        std::cout << "  weights archive: " << archive_path << "\n";
        std::cout << "  golden archive:  " << golden_path << "\n";
        print_tensor_shape("input shape:    ", input_view.shape());
        print_tensor_shape("weight shape:   ", fc_weight.shape());
        print_tensor_shape("bias shape:     ", fc_bias.shape());
        print_tensor_shape("expected shape: ", expected_view.shape());
        std::cout << "  actual shape:   [" << actual_eigen.size() << "]\n";

        if (actual_naive.size() != expected.size() ||
            actual_eigen.size() != expected.size() ||
            actual_default.size() != expected.size()) {
            std::cout << "  shape check:    actual/expected size mismatch\n";
            std::cout << "FAIL\n";
            return 1;
        }

        const float naive_max_abs_error = (actual_naive - expected).cwiseAbs().maxCoeff();
        const float eigen_max_abs_error = (actual_eigen - expected).cwiseAbs().maxCoeff();
        const float default_max_abs_error = (actual_default - expected).cwiseAbs().maxCoeff();
        const float variant_max_abs_delta = (actual_naive - actual_eigen).cwiseAbs().maxCoeff();
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
