#include "dlinf/ops/elementwise.hpp"

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr float kTolerance = 1e-5f;

}  // namespace

int main() {
    const std::size_t size = 1024;

    std::vector<float> dest(size);
    std::vector<float> source(size);

    for (std::size_t i = 0; i < size; ++i) {
        dest[i] = static_cast<float>(i);
        source[i] = static_cast<float>(i * 2);
    }

    // Call elementwise add without shape checks
    dlinf::elementwise_add(dest.data(), source.data(), size);

    float max_abs_error = 0.0f;
    for (std::size_t i = 0; i < size; ++i) {
        const float expected = static_cast<float>(i) + static_cast<float>(i * 2);
        const float abs_err = std::fabs(dest[i] - expected);
        if (abs_err > max_abs_error) {
            max_abs_error = abs_err;
        }
    }

    std::cout << "Elementwise add smoke test\n";
    std::cout << "  size:       " << size << "\n";
    std::cout << "  max error:  " << max_abs_error << "\n";
    std::cout << "  tolerance:  " << kTolerance << "\n";

    if (max_abs_error > kTolerance) {
        std::cout << "FAIL\n";
        return 1;
    }

    std::vector<float> shaped_dest{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> shaped_source{10.0f, 20.0f, 30.0f, 40.0f};

    // Call elementwise_add with shape checks and correct shapes
    dlinf::elementwise_add(
        shaped_dest.data(),
        std::vector<std::uint64_t>{2, 2},
        shaped_source.data(),
        std::vector<std::uint64_t>{2, 2});

    const std::vector<float> shaped_expected{11.0f, 22.0f, 33.0f, 44.0f};
    for (std::size_t i = 0; i < shaped_dest.size(); ++i) {
        const float abs_err = std::fabs(shaped_dest[i] - shaped_expected[i]);
        if (abs_err > max_abs_error) {
            max_abs_error = abs_err;
        }
    }

    bool mismatch_rejected = false;
    // Call elementwise_add with shape checks and mismatches shapes
    try {
        dlinf::elementwise_add(
            shaped_dest.data(),
            std::vector<std::uint64_t>{2, 2},
            shaped_source.data(),
            std::vector<std::uint64_t>{4});
    } catch (const std::invalid_argument&) {
        mismatch_rejected = true;
    }

    std::cout << "  shaped max error: " << max_abs_error << "\n";
    std::cout << "  mismatch rejected: " << (mismatch_rejected ? "yes" : "no") << "\n";

    if (max_abs_error > kTolerance || !mismatch_rejected) {
        std::cout << "FAIL\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}
