#include "dlinf/networks/resnet18.hpp"
#include "dlinf/activation.hpp"
#include "dlinf/blocks/basicblock.hpp"
#include "dlinf/layers/avgpool2d.hpp"
#include "dlinf/layers/batchnorm2d.hpp"
#include "dlinf/layers/conv2d.hpp"
#include "dlinf/layers/linear.hpp"
#include "dlinf/layers/maxpool2d.hpp"

#include <Eigen/Core>
#include <vector>

namespace dlinf {
using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace {

int out_dim(int in, int kernel, int stride, int padding) {
    return (in + 2 * padding - kernel) / stride + 1;
}

RowMatrixXf run_identity_block(
    const std::string& prefix,
    const RowMatrixXf& input,
    const WeightArchive& weights,
    int height,
    int width,
    float bn_epsilon) {

    auto w = [&](const std::string& suffix) {
        return weights.tensor_f32(prefix + "." + suffix);
    };

    const auto out_channels = w("conv1.weight").shape()[0];
    std::vector<float> zb(out_channels, 0.0f);
    TensorViewF32 zero_bias(zb.data(), {out_channels});

    std::vector<float> dummy_storage(1, 0.0f);
    TensorViewF32 dummy(dummy_storage.data(), {1});

    return basicblock_direct(
        input,
        w("conv1.weight"), zero_bias,
        w("bn1.weight"), w("bn1.bias"),
        w("bn1.running_mean"), w("bn1.running_var"),
        w("conv2.weight"), zero_bias,
        w("bn2.weight"), w("bn2.bias"),
        w("bn2.running_mean"), w("bn2.running_var"),
        dummy, dummy, dummy, dummy, dummy, dummy,
        1, 1, height, width, bn_epsilon);
}

RowMatrixXf run_projection_block(
    const std::string& prefix,
    const RowMatrixXf& input,
    const WeightArchive& weights,
    int height,
    int width,
    int stride,
    float bn_epsilon) {

    auto w = [&](const std::string& suffix) {
        return weights.tensor_f32(prefix + "." + suffix);
    };

    const auto out_channels = w("conv1.weight").shape()[0];
    std::vector<float> zb(out_channels, 0.0f);
    TensorViewF32 zero_bias(zb.data(), {out_channels});

    const auto ds_out_channels = w("downsample.0.weight").shape()[0];
    std::vector<float> ds_zb(ds_out_channels, 0.0f);
    TensorViewF32 ds_zero_bias(ds_zb.data(), {ds_out_channels});

    return basicblock_direct(
        input,
        w("conv1.weight"), zero_bias,
        w("bn1.weight"), w("bn1.bias"),
        w("bn1.running_mean"), w("bn1.running_var"),
        w("conv2.weight"), zero_bias,
        w("bn2.weight"), w("bn2.bias"),
        w("bn2.running_mean"), w("bn2.running_var"),
        w("downsample.0.weight"), ds_zero_bias,
        w("downsample.1.weight"), w("downsample.1.bias"),
        w("downsample.1.running_mean"), w("downsample.1.running_var"),
        stride, 1, height, width, bn_epsilon);
}

}  // namespace

Eigen::VectorXf resnet18_direct(
    const Eigen::Ref<const RowMatrixXf>& input,
    const WeightArchive& weights,
    int input_height,
    int input_width) {

    const float bn_epsilon = 1e-5f;
    auto w = [&](const std::string& name) { return weights.tensor_f32(name); };

    // --- stem: conv1 -> bn1 -> relu -> maxpool ---

    std::vector<float> zb(64, 0.0f);
    TensorViewF32 zero_bias(zb.data(), {64});

    auto height = input_height;
    auto width = input_width;

    auto out = conv2d_naive_direct(input, w("conv1.weight"), zero_bias, 2, 3, height, width);
    height = out_dim(height, 7, 2, 3);  // 224 -> 112
    width  = out_dim(width,  7, 2, 3);

    out = batchnorm2d_direct(out, w("bn1.weight"), w("bn1.bias"),
                             w("bn1.running_mean"), w("bn1.running_var"), bn_epsilon);
    relu_inplace(out.data(), out.size());

    out = maxpool2d_direct(out, height, width, 3, 2, 1);
    height = out_dim(height, 3, 2, 1);  // 112 -> 56
    width  = out_dim(width,  3, 2, 1);

    // --- layer1: two identity blocks ---

    out = run_identity_block("layer1.0", out, weights, height, width, bn_epsilon);
    out = run_identity_block("layer1.1", out, weights, height, width, bn_epsilon);

    // --- layer2: projection then identity ---

    out = run_projection_block("layer2.0", out, weights, height, width, 2, bn_epsilon);
    height = out_dim(height, 3, 2, 1);  // 56 -> 28
    width  = out_dim(width,  3, 2, 1);

    out = run_identity_block("layer2.1", out, weights, height, width, bn_epsilon);

    // --- layer3: projection then identity ---

    out = run_projection_block("layer3.0", out, weights, height, width, 2, bn_epsilon);
    height = out_dim(height, 3, 2, 1);  // 28 -> 14
    width  = out_dim(width,  3, 2, 1);

    out = run_identity_block("layer3.1", out, weights, height, width, bn_epsilon);

    // --- layer4: projection then identity ---

    out = run_projection_block("layer4.0", out, weights, height, width, 2, bn_epsilon);
    height = out_dim(height, 3, 2, 1);  // 14 -> 7
    width  = out_dim(width,  3, 2, 1);

    out = run_identity_block("layer4.1", out, weights, height, width, bn_epsilon);

    // --- head: avgpool -> flatten -> fc ---

    out = avgpool2d_direct(out, height, width, 7, 1, 0);

    Eigen::Map<const Eigen::VectorXf> flat(out.data(), out.size());
    return linear_eigen(flat, w("fc.weight"), w("fc.bias"));
}

}  // namespace dlinf
