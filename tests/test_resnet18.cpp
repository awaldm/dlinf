#include "dlinf/networks/resnet18.hpp"
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
        std::cerr << "usage: test_resnet18 <resnet_weights.elw> <resnet18_full_golden.elw>\n";
        return 1;
    }

    const std::string archive_path = argv[1];
    const std::string golden_path = argv[2];

    try {
        const auto archive = dlinf::WeightArchive::load(archive_path);
        const auto golden = dlinf::WeightArchive::load(golden_path);

        const auto input_view = golden.tensor_f32("resnet18.input");
        const auto expected_view = golden.tensor_f32("resnet18.expected");

        const auto input = dlinf::test_support::map_chw_as_rows(input_view);
        const auto expected = dlinf::test_support::map_vector(expected_view);

        const Eigen::VectorXf actual = dlinf::resnet18_direct(input, archive, 224, 224);

        std::cout << "ResNet-18 full model golden test\n";
        std::cout << "  weights archive: " << archive_path << "\n";
        std::cout << "  golden archive:  " << golden_path << "\n";
        dlinf::test_support::print_tensor_shape("input shape:    ", input_view.shape());
        dlinf::test_support::print_tensor_shape("expected shape: ", expected_view.shape());
        std::cout << "  actual size:     [" << actual.size() << "]\n";
        if (actual.size() != expected.size()) {
            std::cout << "  shape check:    actual/expected size mismatch\n";
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
