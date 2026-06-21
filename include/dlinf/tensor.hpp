#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dlinf {

class TensorViewF32 {
public:
    TensorViewF32(const float* data, std::vector<std::uint64_t> shape)
        : data_(data), shape_(std::move(shape)), size_(element_count(shape_)) {}

    [[nodiscard]] const float* data() const noexcept { return data_; }
    [[nodiscard]] const std::vector<std::uint64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    static std::size_t element_count(const std::vector<std::uint64_t>& shape) {
        if (shape.empty()) {
            return 1;
        }
        return std::accumulate(
            shape.begin(),
            shape.end(),
            static_cast<std::uint64_t>(1),
            [](std::uint64_t lhs, std::uint64_t rhs) {
                if (rhs == 0) {
                    throw std::invalid_argument("tensor shape contains a zero dimension");
                }
                return lhs * rhs;
            });
    }

    const float* data_;
    std::vector<std::uint64_t> shape_;
    std::size_t size_;
};

}  // namespace dlinf
