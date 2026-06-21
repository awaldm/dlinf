#include "dlinf/layers/batchnorm2d.hpp"
#include "dlinf/layers/conv2d.hpp"
#include "dlinf/tensor.hpp"
#include "dlinf/weight_archive.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kTolerance = 1e-5f;
constexpr int kConv1Stride = 2;
constexpr int kConv1Padding = 3;

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
        std::cerr << "usage: test_conv_bn <resnet_weights.elw> <conv1_bn1_golden.elw>\n";
        return 1;
    }

    const std::string archive_path = argv[1];
    const std::string golden_path = argv[2];

    try {
        const auto archive = eigen_learn::WeightArchive::load(archive_path);
        const auto golden = eigen_learn::WeightArchive::load(golden_path);

        const auto conv1_weight = archive.tensor_f32("conv1.weight");
        const auto bn1_weight = archive.tensor_f32("bn1.weight");
        const auto bn1_bias = archive.tensor_f32("bn1.bias");
        const auto bn1_running_mean = archive.tensor_f32("bn1.running_mean");
        const auto bn1_running_var = archive.tensor_f32("bn1.running_var");
        const auto input_view = golden.tensor_f32("conv1_bn1.input");
        const auto expected_view = golden.tensor_f32("conv1_bn1.expected");

        const auto out_channels = conv1_weight.shape()[0];
        std::vector<float> zero_bias(out_channels, 0.0f);
        eigen_learn::TensorViewF32 conv1_bias(zero_bias.data(), {out_channels});

        Eigen::Map<const RowMatrixXf> input(
            input_view.data(),
            static_cast<Eigen::Index>(input_view.shape()[0]),
            static_cast<Eigen::Index>(input_view.shape()[1] * input_view.shape()[2]));
        Eigen::Map<const RowMatrixXf> expected(
            expected_view.data(),
            static_cast<Eigen::Index>(expected_view.shape()[0]),
            static_cast<Eigen::Index>(expected_view.shape()[1] * expected_view.shape()[2]));

        const auto height = static_cast<int>(input_view.shape()[1]);
        const auto width = static_cast<int>(input_view.shape()[2]);
        const RowMatrixXf conv = eigen_learn::conv2d_naive_direct(
            input,
            conv1_weight,
            conv1_bias,
            kConv1Stride,
            kConv1Padding,
            height,
            width);
        const RowMatrixXf conv_im2col = eigen_learn::conv2d_im2col_eigen(
            input,
            conv1_weight,
            conv1_bias,
            kConv1Stride,
            kConv1Padding,
            height,
            width);
        const RowMatrixXf actual_direct = eigen_learn::batchnorm2d_direct(
            conv,
            bn1_weight,
            bn1_bias,
            bn1_running_mean,
            bn1_running_var,
            1e-5f);
        const RowMatrixXf actual_im2col = eigen_learn::batchnorm2d_direct(
            conv_im2col,
            bn1_weight,
            bn1_bias,
            bn1_running_mean,
            bn1_running_var,
            1e-5f);
        const RowMatrixXf actual_default = eigen_learn::batchnorm2d(
            conv,
            bn1_weight,
            bn1_bias,
            bn1_running_mean,
            bn1_running_var,
            1e-5f);

        std::cout << "Conv2D + BatchNorm2D golden test\n";
        std::cout << "  weights archive: " << archive_path << "\n";
        std::cout << "  golden archive:  " << golden_path << "\n";
        print_tensor_shape("input shape:    ", input_view.shape());
        print_tensor_shape("conv weight shape: ", conv1_weight.shape());
        print_tensor_shape("bn weight shape:   ", bn1_weight.shape());
        print_tensor_shape("bn bias shape:     ", bn1_bias.shape());
        print_tensor_shape("expected tensor shape: ", expected_view.shape());
        std::cout << "  conv comparison view:     [" << conv.rows() << " x " << conv.cols() << "]\n";
        std::cout << "  expected comparison view: [" << expected.rows() << " x " << expected.cols() << "]\n";
        std::cout << "  actual comparison view:   [" << actual_direct.rows() << " x " << actual_direct.cols() << "]\n";

        if (actual_direct.rows() != expected.rows() || actual_direct.cols() != expected.cols() ||
            actual_im2col.rows() != expected.rows() || actual_im2col.cols() != expected.cols() ||
            actual_default.rows() != expected.rows() || actual_default.cols() != expected.cols()) {
            std::cout << "  shape check:    actual/expected shape mismatch\n";
            std::cout << "FAIL\n";
            return 1;
        }

        const float direct_max_abs_error = (actual_direct - expected).cwiseAbs().maxCoeff();
        const float im2col_max_abs_error = (actual_im2col - expected).cwiseAbs().maxCoeff();
        const float default_max_abs_error = (actual_default - expected).cwiseAbs().maxCoeff();
        const float direct_im2col_max_abs_delta = (actual_direct - actual_im2col).cwiseAbs().maxCoeff();
        const float direct_default_max_abs_delta = (actual_direct - actual_default).cwiseAbs().maxCoeff();
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
