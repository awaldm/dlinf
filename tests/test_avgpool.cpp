#include "dlinf/layers/avgpool2d.hpp"
#include "dlinf/weight_archive.hpp"
#include "test_support.hpp"

#include <Eigen/Core>

#include <exception>
#include <iostream>
#include <string>

namespace {

using RowMatrixXf = dlinf::test_support::RowMatrixXf;

constexpr float kTolerance = 1e-5f;
constexpr int kKernelSize = 7;
constexpr int kStride = 1;
constexpr int kPadding = 0;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_avgpool <avgpool_golden.elw>\n";
        return 1;
    }

    const std::string golden_path = argv[1];

    try {
        const auto golden = dlinf::WeightArchive::load(golden_path);

        const auto input_view = golden.tensor_f32("avgpool.input");
        const auto expected_view = golden.tensor_f32("avgpool.expected");

        const auto input = dlinf::test_support::map_chw_as_rows(input_view);
        const auto expected = dlinf::test_support::map_chw_as_rows(expected_view);

        const auto height = static_cast<int>(input_view.shape()[1]);
        const auto width = static_cast<int>(input_view.shape()[2]);

        const RowMatrixXf actual = dlinf::avgpool2d_direct(
            input, height, width, kKernelSize, kStride, kPadding);

        std::cout << "AvgPool2D golden test\n";
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
