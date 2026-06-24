#include "dlinf/blocks/basicblock.hpp"
#include "dlinf/layers/avgpool2d.hpp"
#include "dlinf/layers/batchnorm2d.hpp"
#include "dlinf/layers/conv2d.hpp"
#include "dlinf/layers/linear.hpp"
#include "dlinf/layers/maxpool2d.hpp"
#include "dlinf/weight_archive.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#ifndef DLINF_CXXFLAGS
#define DLINF_CXXFLAGS "unknown"
#endif

namespace {

constexpr float kTolerance = 2e-5f;
constexpr int kConv1Stride = 2;
constexpr int kConv1Padding = 3;

using Clock = std::chrono::steady_clock;
using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Options {
    std::string weights_path = "artifacts/resnet18/resnet18_imagenet1k_v1.elw";
    std::string fc_golden_path = "artifacts/resnet18/fc_golden.elw";
    std::string conv_golden_path = "artifacts/resnet18/conv1_golden.elw";
    std::string conv_bn_golden_path = "artifacts/resnet18/conv1_bn1_golden.elw";
    std::string basicblock_golden_path = "artifacts/resnet18/layer1_0_basicblock_golden.elw";
    std::string maxpool_golden_path = "artifacts/resnet18/maxpool_golden.elw";
    std::string projection_basicblock_golden_path = "artifacts/resnet18/layer2_0_projection_basicblock_golden.elw";
    std::string avgpool_golden_path = "artifacts/resnet18/avgpool_golden.elw";
    int warmup = 2;
    int iterations = 10;
};

struct BenchmarkStats {
    double median_ms;
    double p10_ms;
    double p90_ms;
    double checksum;
};

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
                break;
        }
    }
    return out.str();
}

std::string shape_json(const std::vector<std::uint64_t>& shape) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

std::string host_name() {
    char buffer[256] = {};
    if (gethostname(buffer, sizeof(buffer) - 1) != 0) {
        return "unknown";
    }
    return buffer;
}

std::string cpu_model() {
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line)) {
        const std::string key = "model name";
        if (line.rfind(key, 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                auto value = line.substr(colon + 1);
                const auto first = value.find_first_not_of(" \t");
                if (first != std::string::npos) {
                    value.erase(0, first);
                }
                return value;
            }
        }
    }
    return "unknown";
}

double percentile(const std::vector<double>& sorted_values, double q) {
    if (sorted_values.empty()) {
        throw std::runtime_error("cannot compute percentile of empty sample");
    }
    const auto index = static_cast<std::size_t>(
        std::floor(q * static_cast<double>(sorted_values.size() - 1)));
    return sorted_values[index];
}

BenchmarkStats run_benchmark(
    int warmup,
    int iterations,
    const std::function<double()>& work) {
    if (warmup < 0 || iterations <= 0) {
        throw std::runtime_error("warmup must be >= 0 and iterations must be > 0");
    }

    double checksum = 0.0;
    for (int i = 0; i < warmup; ++i) {
        checksum += work();
    }

    std::vector<double> durations_ms;
    durations_ms.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        checksum += work();
        const auto stop = Clock::now();
        durations_ms.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }

    std::sort(durations_ms.begin(), durations_ms.end());
    return {
        percentile(durations_ms, 0.5),
        percentile(durations_ms, 0.1),
        percentile(durations_ms, 0.9),
        checksum,
    };
}

void print_record(
    const std::string& op,
    const std::string& impl,
    const std::vector<std::uint64_t>& input_shape,
    const std::vector<std::uint64_t>& weight_shape,
    int warmup,
    int iterations,
    double max_abs_error,
    const BenchmarkStats& stats,
    const std::string& host,
    const std::string& cpu) {
    std::cout << std::fixed << std::setprecision(6)
              << "{"
              << "\"op\":\"" << json_escape(op) << "\","
              << "\"impl\":\"" << json_escape(impl) << "\","
              << "\"precision\":\"fp32\","
              << "\"threads\":1,"
              << "\"warmup\":" << warmup << ","
              << "\"iterations\":" << iterations << ","
              << "\"median_ms\":" << stats.median_ms << ","
              << "\"p10_ms\":" << stats.p10_ms << ","
              << "\"p90_ms\":" << stats.p90_ms << ","
              << "\"max_abs_error\":" << max_abs_error << ","
              << "\"checksum\":" << stats.checksum << ","
              << "\"input_shape\":" << shape_json(input_shape) << ","
              << "\"weight_shape\":" << shape_json(weight_shape) << ","
              << "\"compiler\":\"" << json_escape(__VERSION__) << "\","
              << "\"compiler_flags\":\"" << json_escape(DLINF_CXXFLAGS) << "\","
              << "\"host\":\"" << json_escape(host) << "\","
              << "\"cpu_model\":\"" << json_escape(cpu) << "\""
              << "}\n";
}

void require_close(const std::string& label, double max_abs_error) {
    if (max_abs_error > kTolerance) {
        std::ostringstream message;
        message << label << " failed correctness check: max_abs_error="
                << max_abs_error << " tolerance=" << kTolerance;
        throw std::runtime_error(message.str());
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--weights") {
            options.weights_path = require_value(arg);
        } else if (arg == "--fc-golden") {
            options.fc_golden_path = require_value(arg);
        } else if (arg == "--conv-golden") {
            options.conv_golden_path = require_value(arg);
        } else if (arg == "--conv-bn-golden") {
            options.conv_bn_golden_path = require_value(arg);
        } else if (arg == "--basicblock-golden") {
            options.basicblock_golden_path = require_value(arg);
        } else if (arg == "--maxpool-golden") {
            options.maxpool_golden_path = require_value(arg);
        } else if (arg == "--projection-basicblock-golden") {
            options.projection_basicblock_golden_path = require_value(arg);
        } else if (arg == "--avgpool-golden") {
            options.avgpool_golden_path = require_value(arg);
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(require_value(arg));
        } else if (arg == "--iterations") {
            options.iterations = std::stoi(require_value(arg));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: bench_kernels [--weights PATH] [--fc-golden PATH] "
                   "[--conv-golden PATH] [--conv-bn-golden PATH] "
                   "[--warmup N] [--iterations N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

void benchmark_linear(
    const Options& options,
    const dlinf::WeightArchive& weights,
    const std::string& host,
    const std::string& cpu) {
    const auto golden = dlinf::WeightArchive::load(options.fc_golden_path);
    const auto fc_weight = weights.tensor_f32("fc.weight");
    const auto fc_bias = weights.tensor_f32("fc.bias");
    const auto input_view = golden.tensor_f32("fc.input");
    const auto expected_view = golden.tensor_f32("fc.expected");

    Eigen::Map<const Eigen::VectorXf> input(input_view.data(), input_view.size());
    Eigen::Map<const Eigen::VectorXf> expected(expected_view.data(), expected_view.size());

    const auto run_case = [&](const std::string& impl, const auto& fn) {
        const Eigen::VectorXf actual = fn();
        const double max_abs_error = (actual - expected).cwiseAbs().maxCoeff();
        require_close("linear_fc/" + impl, max_abs_error);

        const BenchmarkStats stats = run_benchmark(options.warmup, options.iterations, [&]() {
            const Eigen::VectorXf output = fn();
            return static_cast<double>(output.sum());
        });
        print_record(
            "linear_fc",
            impl,
            input_view.shape(),
            fc_weight.shape(),
            options.warmup,
            options.iterations,
            max_abs_error,
            stats,
            host,
            cpu);
    };

    run_case("linear_naive", [&]() {
        return dlinf::linear_naive(input, fc_weight, fc_bias);
    });
    run_case("linear_eigen", [&]() {
        return dlinf::linear_eigen(input, fc_weight, fc_bias);
    });
}

void benchmark_conv1(
    const Options& options,
    const dlinf::WeightArchive& weights,
    const std::string& host,
    const std::string& cpu) {
    const auto golden = dlinf::WeightArchive::load(options.conv_golden_path);
    const auto conv1_weight = weights.tensor_f32("conv1.weight");
    const auto input_view = golden.tensor_f32("conv1.input");
    const auto expected_view = golden.tensor_f32("conv1.expected");

    const auto out_channels = conv1_weight.shape()[0];
    std::vector<float> zero_bias(out_channels, 0.0f);
    const dlinf::TensorViewF32 conv1_bias(zero_bias.data(), {out_channels});

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

    const auto run_case = [&](const std::string& impl, const auto& fn) {
        const RowMatrixXf actual = fn();
        const double max_abs_error = (actual - expected).cwiseAbs().maxCoeff();
        require_close("conv1/" + impl, max_abs_error);

        const BenchmarkStats stats = run_benchmark(options.warmup, options.iterations, [&]() {
            const RowMatrixXf output = fn();
            return static_cast<double>(output.sum());
        });
        print_record(
            "conv1",
            impl,
            input_view.shape(),
            conv1_weight.shape(),
            options.warmup,
            options.iterations,
            max_abs_error,
            stats,
            host,
            cpu);
    };

    run_case("conv2d_naive_direct", [&]() {
        return dlinf::conv2d_naive_direct(
            input, conv1_weight, conv1_bias, kConv1Stride, kConv1Padding, height, width);
    });
    run_case("conv2d_im2col_eigen", [&]() {
        return dlinf::conv2d_im2col_eigen(
            input, conv1_weight, conv1_bias, kConv1Stride, kConv1Padding, height, width);
    });
}

void benchmark_conv1_bn1(
    const Options& options,
    const dlinf::WeightArchive& weights,
    const std::string& host,
    const std::string& cpu) {
    const auto golden = dlinf::WeightArchive::load(options.conv_bn_golden_path);
    const auto conv1_weight = weights.tensor_f32("conv1.weight");
    const auto bn1_weight = weights.tensor_f32("bn1.weight");
    const auto bn1_bias = weights.tensor_f32("bn1.bias");
    const auto bn1_running_mean = weights.tensor_f32("bn1.running_mean");
    const auto bn1_running_var = weights.tensor_f32("bn1.running_var");
    const auto input_view = golden.tensor_f32("conv1_bn1.input");
    const auto expected_view = golden.tensor_f32("conv1_bn1.expected");

    const auto out_channels = conv1_weight.shape()[0];
    std::vector<float> zero_bias(out_channels, 0.0f);
    const dlinf::TensorViewF32 conv1_bias(zero_bias.data(), {out_channels});

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

    const auto batchnorm = [&](const RowMatrixXf& conv) {
        return dlinf::batchnorm2d_direct(
            conv,
            bn1_weight,
            bn1_bias,
            bn1_running_mean,
            bn1_running_var,
            1e-5f);
    };

    const auto run_case = [&](const std::string& impl, const auto& conv_fn) {
        const RowMatrixXf actual = batchnorm(conv_fn());
        const double max_abs_error = (actual - expected).cwiseAbs().maxCoeff();
        require_close("conv1_bn1/" + impl, max_abs_error);

        const BenchmarkStats stats = run_benchmark(options.warmup, options.iterations, [&]() {
            const RowMatrixXf output = batchnorm(conv_fn());
            return static_cast<double>(output.sum());
        });
        print_record(
            "conv1_bn1",
            impl,
            input_view.shape(),
            conv1_weight.shape(),
            options.warmup,
            options.iterations,
            max_abs_error,
            stats,
            host,
            cpu);
    };

    run_case("conv2d_naive_direct_batchnorm2d_direct", [&]() {
        return dlinf::conv2d_naive_direct(
            input, conv1_weight, conv1_bias, kConv1Stride, kConv1Padding, height, width);
    });
    run_case("conv2d_im2col_eigen_batchnorm2d_direct", [&]() {
        return dlinf::conv2d_im2col_eigen(
            input, conv1_weight, conv1_bias, kConv1Stride, kConv1Padding, height, width);
    });
}

void benchmark_basicblock(
    const Options& options,
    const dlinf::WeightArchive& weights,
    const std::string& host,
    const std::string& cpu) {

    const auto golden = dlinf::WeightArchive::load(options.basicblock_golden_path);

    const auto conv1_weight = weights.tensor_f32("layer1.0.conv1.weight");
    const auto bn1_weight = weights.tensor_f32("layer1.0.bn1.weight");
    const auto bn1_bias = weights.tensor_f32("layer1.0.bn1.bias");
    const auto bn1_running_mean = weights.tensor_f32("layer1.0.bn1.running_mean");
    const auto bn1_running_var = weights.tensor_f32("layer1.0.bn1.running_var");
    const auto conv2_weight = weights.tensor_f32("layer1.0.conv2.weight");
    const auto bn2_weight = weights.tensor_f32("layer1.0.bn2.weight");
    const auto bn2_bias = weights.tensor_f32("layer1.0.bn2.bias");
    const auto bn2_running_mean = weights.tensor_f32("layer1.0.bn2.running_mean");
    const auto bn2_running_var = weights.tensor_f32("layer1.0.bn2.running_var");

    const auto out_channels = conv1_weight.shape()[0];
    std::vector<float> zero_bias(out_channels, 0.0f);
    const dlinf::TensorViewF32 conv1_bias(zero_bias.data(), {out_channels});
    const dlinf::TensorViewF32 conv2_bias(zero_bias.data(), {out_channels});

    const auto input_view = golden.tensor_f32("layer1.0.input");
    const auto expected_view = golden.tensor_f32("layer1.0.expected");

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
    const int stride = 1;
    const int padding = 1;
    const float bn_epsilon = 1e-5f;

    // Identity block: dummy downsample tensors (never entered)
    std::vector<float> dummy_storage(1, 0.0f);
    const dlinf::TensorViewF32 dummy(dummy_storage.data(), {1});

    const auto run_case = [&](const std::string& impl, const auto& fn) {
        const RowMatrixXf actual = fn();
        const double max_abs_error = (actual - expected).cwiseAbs().maxCoeff();
        require_close("layer1.0/" + impl, max_abs_error);

        const BenchmarkStats stats = run_benchmark(options.warmup, options.iterations, [&]() {
            const RowMatrixXf output = fn();
            return static_cast<double>(output.sum());
        });
        print_record(
            "layer1.0",
            impl,
            input_view.shape(),
            conv1_weight.shape(),
            options.warmup,
            options.iterations,
            max_abs_error,
            stats,
            host,
            cpu);
    };

    run_case("basicblock_direct", [&]() {
        return dlinf::basicblock_direct(
            input, conv1_weight, conv1_bias,
            bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
            conv2_weight, conv2_bias,
            bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
            dummy, dummy, dummy, dummy, dummy, dummy,
            stride, padding, height, width, bn_epsilon);
    });
}

void benchmark_maxpool(
    const Options& options,
    const dlinf::WeightArchive& /*weights*/,
    const std::string& host,
    const std::string& cpu) {
    const auto golden = dlinf::WeightArchive::load(options.maxpool_golden_path);

    const auto input_view = golden.tensor_f32("maxpool.input");
    const auto expected_view = golden.tensor_f32("maxpool.expected");

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
    constexpr int kernel_size = 3;
    constexpr int stride = 2;
    constexpr int padding = 1;

    const auto run_case = [&](const std::string& impl, const auto& fn) {
        const RowMatrixXf actual = fn();
        const double max_abs_error = (actual - expected).cwiseAbs().maxCoeff();
        require_close("maxpool/" + impl, max_abs_error);

        const BenchmarkStats stats = run_benchmark(options.warmup, options.iterations, [&]() {
            const RowMatrixXf output = fn();
            return static_cast<double>(output.sum());
        });
        print_record(
            "maxpool",
            impl,
            input_view.shape(),
            {},
            options.warmup,
            options.iterations,
            max_abs_error,
            stats,
            host,
            cpu);
    };

    run_case("maxpool2d_direct", [&]() {
        return dlinf::maxpool2d_direct(input, height, width, kernel_size, stride, padding);
    });
}

void benchmark_projection_basicblock(
    const Options& options,
    const dlinf::WeightArchive& weights,
    const std::string& host,
    const std::string& cpu) {

    const auto golden = dlinf::WeightArchive::load(options.projection_basicblock_golden_path);

    const auto conv1_weight = weights.tensor_f32("layer2.0.conv1.weight");
    const auto bn1_weight = weights.tensor_f32("layer2.0.bn1.weight");
    const auto bn1_bias = weights.tensor_f32("layer2.0.bn1.bias");
    const auto bn1_running_mean = weights.tensor_f32("layer2.0.bn1.running_mean");
    const auto bn1_running_var = weights.tensor_f32("layer2.0.bn1.running_var");
    const auto conv2_weight = weights.tensor_f32("layer2.0.conv2.weight");
    const auto bn2_weight = weights.tensor_f32("layer2.0.bn2.weight");
    const auto bn2_bias = weights.tensor_f32("layer2.0.bn2.bias");
    const auto bn2_running_mean = weights.tensor_f32("layer2.0.bn2.running_mean");
    const auto bn2_running_var = weights.tensor_f32("layer2.0.bn2.running_var");
    const auto ds_conv_weight = weights.tensor_f32("layer2.0.downsample.0.weight");
    const auto ds_bn_weight = weights.tensor_f32("layer2.0.downsample.1.weight");
    const auto ds_bn_bias = weights.tensor_f32("layer2.0.downsample.1.bias");
    const auto ds_bn_running_mean = weights.tensor_f32("layer2.0.downsample.1.running_mean");
    const auto ds_bn_running_var = weights.tensor_f32("layer2.0.downsample.1.running_var");

    const auto out_channels = conv1_weight.shape()[0];
    std::vector<float> zero_bias(out_channels, 0.0f);
    const dlinf::TensorViewF32 conv1_bias(zero_bias.data(), {out_channels});
    const dlinf::TensorViewF32 conv2_bias(zero_bias.data(), {out_channels});

    const auto ds_out_channels = ds_conv_weight.shape()[0];
    std::vector<float> ds_zero_bias(ds_out_channels, 0.0f);
    const dlinf::TensorViewF32 ds_conv_bias(ds_zero_bias.data(), {ds_out_channels});

    const auto input_view = golden.tensor_f32("layer2.0.input");
    const auto expected_view = golden.tensor_f32("layer2.0.expected");

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
    const int stride = 2;
    const int padding = 1;
    const float bn_epsilon = 1e-5f;

    const auto run_case = [&](const std::string& impl, const auto& fn) {
        const RowMatrixXf actual = fn();
        const double max_abs_error = (actual - expected).cwiseAbs().maxCoeff();
        require_close("layer2.0/" + impl, max_abs_error);

        const BenchmarkStats stats = run_benchmark(options.warmup, options.iterations, [&]() {
            const RowMatrixXf output = fn();
            return static_cast<double>(output.sum());
        });
        print_record(
            "layer2.0",
            impl,
            input_view.shape(),
            conv1_weight.shape(),
            options.warmup,
            options.iterations,
            max_abs_error,
            stats,
            host,
            cpu);
    };

    run_case("basicblock_direct", [&]() {
        return dlinf::basicblock_direct(
            input, conv1_weight, conv1_bias,
            bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
            conv2_weight, conv2_bias,
            bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
            ds_conv_weight, ds_conv_bias,
            ds_bn_weight, ds_bn_bias, ds_bn_running_mean, ds_bn_running_var,
            stride, padding, height, width, bn_epsilon);
    });
}

void benchmark_avgpool(
    const Options& options,
    const dlinf::WeightArchive& /*weights*/,
    const std::string& host,
    const std::string& cpu) {
    const auto golden = dlinf::WeightArchive::load(options.avgpool_golden_path);

    const auto input_view = golden.tensor_f32("avgpool.input");
    const auto expected_view = golden.tensor_f32("avgpool.expected");

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
    constexpr int kernel_size = 7;
    constexpr int stride = 1;
    constexpr int padding = 0;

    const auto run_case = [&](const std::string& impl, const auto& fn) {
        const RowMatrixXf actual = fn();
        const double max_abs_error = (actual - expected).cwiseAbs().maxCoeff();
        require_close("avgpool/" + impl, max_abs_error);

        const BenchmarkStats stats = run_benchmark(options.warmup, options.iterations, [&]() {
            const RowMatrixXf output = fn();
            return static_cast<double>(output.sum());
        });
        print_record(
            "avgpool",
            impl,
            input_view.shape(),
            {},
            options.warmup,
            options.iterations,
            max_abs_error,
            stats,
            host,
            cpu);
    };

    run_case("avgpool2d_direct", [&]() {
        return dlinf::avgpool2d_direct(input, height, width, kernel_size, stride, padding);
    });
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto weights = dlinf::WeightArchive::load(options.weights_path);
        const std::string host = host_name();
        const std::string cpu = cpu_model();

        benchmark_linear(options, weights, host, cpu);
        benchmark_conv1(options, weights, host, cpu);
        benchmark_conv1_bn1(options, weights, host, cpu);
        benchmark_basicblock(options, weights, host, cpu);
        benchmark_maxpool(options, weights, host, cpu);
        benchmark_projection_basicblock(options, weights, host, cpu);
        benchmark_avgpool(options, weights, host, cpu);

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "bench_kernels failed: " << exc.what() << "\n";
        return 1;
    }
}
