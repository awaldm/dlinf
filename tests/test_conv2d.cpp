#include "dlinf/layers/conv2d.hpp"
#include "dlinf/tensor.hpp"
#include "dlinf/weight_archive.hpp"
#include "test_support.hpp"

#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <string>

namespace {

constexpr float kTolerance = 1e-5f;

using RowMatrixXf = dlinf::test_support::RowMatrixXf;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_conv2d <resnet_weights.elw> <conv1_golden.elw>\n";
        return 1;
    }

    const std::string archive_path = argv[1];
    const std::string golden_path = argv[2];

    const int stride = 2;
    const int padding = 3;

    try {
        const auto archive = dlinf::WeightArchive::load(archive_path);
        const auto golden = dlinf::WeightArchive::load(golden_path);

        const auto conv1_weight = archive.tensor_f32("conv1.weight");
        const auto out_channels = conv1_weight.shape()[0]; // get the number of channels via NCHW from weights
        const auto zero_bias = dlinf::test_support::zero_bias_storage(out_channels);
        const auto conv1_bias = dlinf::test_support::bias_view(zero_bias);
        const auto input_view = golden.tensor_f32("conv1.input");
        const auto expected_view = golden.tensor_f32("conv1.expected");
        
        const auto input = dlinf::test_support::map_chw_as_rows(input_view);
        const auto expected = dlinf::test_support::map_chw_as_rows(expected_view);
        const auto height = input_view.shape()[1];
        const auto width = input_view.shape()[2];
        const RowMatrixXf actual_direct =
            dlinf::conv2d_naive_direct(input, conv1_weight, conv1_bias, stride, padding, height, width);
        const RowMatrixXf actual_im2col =
            dlinf::conv2d_im2col_eigen(input, conv1_weight, conv1_bias, stride, padding, height, width);
        const RowMatrixXf actual_default =
            dlinf::conv2d(input, conv1_weight, conv1_bias, stride, padding, height, width);

        std::cout << "Conv2D golden test\n";
        std::cout << "  weights archive: " << archive_path << "\n";
        std::cout << "  golden archive:  " << golden_path << "\n";
        dlinf::test_support::print_tensor_shape("input shape:    ", input_view.shape());
        dlinf::test_support::print_tensor_shape("weight shape:   ", conv1_weight.shape());
        std::cout << "  stride:         " << stride << "\n";
        std::cout << "  padding:        " << padding << "\n";
        dlinf::test_support::print_tensor_shape("expected tensor shape: ", expected_view.shape());
        std::cout << "  expected comparison view: [" << expected.rows() << " x " << expected.cols() << "]\n";
        std::cout << "  actual comparison view:   [" << actual_direct.rows() << " x " << actual_direct.cols() << "]\n";

        if (actual_direct.rows() != expected.rows() || actual_direct.cols() != expected.cols() ||
            actual_im2col.rows() != expected.rows() || actual_im2col.cols() != expected.cols() ||
            actual_default.rows() != expected.rows() || actual_default.cols() != expected.cols()) {
            std::cout << "  shape check:    actual/expected shape mismatch\n";
            std::cout << "FAIL\n";
            return 1;
        }

        const float direct_max_abs_error = dlinf::test_support::max_abs_error(actual_direct, expected);
        const float im2col_max_abs_error = dlinf::test_support::max_abs_error(actual_im2col, expected);
        const float default_max_abs_error = dlinf::test_support::max_abs_error(actual_default, expected);
        const float direct_im2col_max_abs_delta = dlinf::test_support::max_abs_error(actual_direct, actual_im2col);
        const float direct_default_max_abs_delta = dlinf::test_support::max_abs_error(actual_direct, actual_default);
        std::cout << "  direct max abs error:  " << direct_max_abs_error << "\n";
        std::cout << "  im2col max abs error:  " << im2col_max_abs_error << "\n";
        std::cout << "  default max abs error: " << default_max_abs_error << "\n";
        std::cout << "  direct/im2col delta:   " << direct_im2col_max_abs_delta << "\n";
        std::cout << "  direct/default delta:  " << direct_default_max_abs_delta << "\n";
        std::cout << "  tolerance:      " << kTolerance << "\n";

        if (direct_max_abs_error > kTolerance ||
            im2col_max_abs_error > kTolerance ||
            default_max_abs_error > kTolerance ||
            direct_im2col_max_abs_delta > kTolerance ||
            direct_default_max_abs_delta > kTolerance) {
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
