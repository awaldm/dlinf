#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace dlinf {

namespace detail {

inline std::size_t element_count(const std::vector<std::uint64_t>& shape) {
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

}  // namespace detail
/**
* Elementwise add. In-place implementation (input destination data gets overwritten, watch out when debugging)
*
* @param source Source data
* @param dest Destination data
*/
inline void elementwise_add(float* dest, const float* source, std::size_t size) {
    Eigen::Map<const Eigen::ArrayXf> source_data(source, static_cast<Eigen::Index>(size));
    Eigen::Map<Eigen::ArrayXf> destination_data(dest, static_cast<Eigen::Index>(size));
    destination_data += source_data;
}

/**
* Elementwise add overloaded with shape checks, now enforcing equal shapes for source and dest.
*
* @param source Source data
* @param dest Destination data
*/
inline void elementwise_add(
    float* dest,
    const std::vector<std::uint64_t>& dest_shape,
    const float* source,
    const std::vector<std::uint64_t>& source_shape) {
    if (dest_shape != source_shape) {
        throw std::invalid_argument("elementwise_add requires identical tensor shapes");
    }
    elementwise_add(dest, source, detail::element_count(dest_shape));
}

}  // namespace dlinf
