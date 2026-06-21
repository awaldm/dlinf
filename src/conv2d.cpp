#include "dlinf/layers/conv2d.hpp"

#include <stdexcept>

namespace eigen_learn {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace {

struct Conv2dShape {
    Eigen::Index C_out;
    Eigen::Index C_in;
    Eigen::Index kernel_height;
    Eigen::Index kernel_width;
    Eigen::Index H_out;
    Eigen::Index W_out;
};

Conv2dShape validate_conv2d_inputs(
    const Eigen::Ref<const RowMatrixXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias,
    int stride,
    int padding,
    int height,
    int width) {
    if (weight.shape().size() != 4) {
        throw std::runtime_error("Conv2D weight must be rank 4");
    }
    if (bias.shape().size() != 1) {
        throw std::runtime_error("Conv2D bias must be rank 1");
    }
    if (stride <= 0) {
        throw std::runtime_error("Conv2D stride must be positive");
    }
    if (padding < 0) {
        throw std::runtime_error("Conv2D padding must be non-negative");
    }
    if (height <= 0 || width <= 0) {
        throw std::runtime_error("Conv2D input height and width must be positive");
    }

    const auto C_out = static_cast<Eigen::Index>(weight.shape()[0]);
    const auto C_in = static_cast<Eigen::Index>(weight.shape()[1]);
    const auto kernel_height = static_cast<Eigen::Index>(weight.shape()[2]);
    const auto kernel_width = static_cast<Eigen::Index>(weight.shape()[3]);

    if (input.rows() != C_in) {
        throw std::runtime_error("Conv2D input channel mismatch");
    }
    if (input.cols() != static_cast<Eigen::Index>(height) * static_cast<Eigen::Index>(width)) {
        throw std::runtime_error("Conv2D flattened input size mismatch");
    }
    if (static_cast<Eigen::Index>(bias.size()) != C_out) {
        throw std::runtime_error("Conv2D bias size mismatch");
    }
    if (kernel_height <= 0 || kernel_width <= 0) {
        throw std::runtime_error("Conv2D kernel dimensions must be positive");
    }

    const Eigen::Index H_out = (height + 2 * padding - kernel_height) / stride + 1;
    const Eigen::Index W_out = (width + 2 * padding - kernel_width) / stride + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("Conv2D output shape is empty");
    }

    return {C_out, C_in, kernel_height, kernel_width, H_out, W_out};
}

}  // namespace

RowMatrixXf conv2d_naive_direct(
    const Eigen::Ref<const RowMatrixXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias,
    int stride,
    int padding,
    int height,
    int width) {
    const Conv2dShape shape =
        validate_conv2d_inputs(input, weight, bias, stride, padding, height, width);

    RowMatrixXf output(shape.C_out, shape.H_out * shape.W_out);

    for (Eigen::Index oc = 0; oc < shape.C_out; ++oc) {
        for (Eigen::Index oy = 0; oy < shape.H_out; ++oy) {
            for (Eigen::Index ox = 0; ox < shape.W_out; ++ox) {
                float sum = bias.data()[oc];
                for (Eigen::Index ic = 0; ic < shape.C_in; ++ic) {
                    for (Eigen::Index ky = 0; ky < shape.kernel_height; ++ky) {
                        for (Eigen::Index kx = 0; kx < shape.kernel_width; ++kx) {
                            const Eigen::Index iy = oy * stride + ky - padding;
                            const Eigen::Index ix = ox * stride + kx - padding;
                            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                                const float w =
                                    weight.data()[((oc * shape.C_in + ic) * shape.kernel_height + ky) *
                                                      shape.kernel_width +
                                                  kx];
                                sum += input(ic, iy * width + ix) * w;
                            }
                        }
                    }
                }
                output(oc, oy * shape.W_out + ox) = sum;
            }
        }
    }

    return output;
}

RowMatrixXf conv2d_im2col_eigen(
    const Eigen::Ref<const RowMatrixXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias,
    int stride,
    int padding,
    int height,
    int width) {
    const Conv2dShape shape =
        validate_conv2d_inputs(input, weight, bias, stride, padding, height, width);
    const Eigen::Index kernel_elements = shape.C_in * shape.kernel_height * shape.kernel_width;
    const Eigen::Index output_elements = shape.H_out * shape.W_out;

    RowMatrixXf columns(kernel_elements, output_elements);
    for (Eigen::Index oy = 0; oy < shape.H_out; ++oy) {
        for (Eigen::Index ox = 0; ox < shape.W_out; ++ox) {
            const Eigen::Index col = oy * shape.W_out + ox;
            for (Eigen::Index ic = 0; ic < shape.C_in; ++ic) {
                for (Eigen::Index ky = 0; ky < shape.kernel_height; ++ky) {
                    for (Eigen::Index kx = 0; kx < shape.kernel_width; ++kx) {
                        const Eigen::Index row =
                            (ic * shape.kernel_height + ky) * shape.kernel_width + kx;
                        const Eigen::Index iy = oy * stride + ky - padding;
                        const Eigen::Index ix = ox * stride + kx - padding;
                        columns(row, col) =
                            (iy >= 0 && iy < height && ix >= 0 && ix < width)
                                ? input(ic, iy * width + ix)
                                : 0.0f;
                    }
                }
            }
        }
    }

    Eigen::Map<const RowMatrixXf> weights(weight.data(), shape.C_out, kernel_elements);
    Eigen::Map<const Eigen::VectorXf> b(bias.data(), shape.C_out);

    RowMatrixXf output = weights * columns;
    output.colwise() += b;
    return output;
}

RowMatrixXf conv2d(
    const Eigen::Ref<const RowMatrixXf>& input,
    const TensorViewF32& weight,
    const TensorViewF32& bias,
    int stride,
    int padding,
    int height,
    int width) {
    return conv2d_naive_direct(input, weight, bias, stride, padding, height, width);
}

}  // namespace eigen_learn
