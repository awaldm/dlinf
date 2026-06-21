#pragma once

#include "dlinf/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dlinf {

enum class DType : std::uint32_t {
    Float32 = 1,
    Float64 = 2,
    Int64 = 3,
    Int32 = 4,
    UInt8 = 5,
    Float16 = 6,
};

struct TensorRecordMeta {
    std::string name;
    DType dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset;
    std::uint64_t nbytes;
    std::uint32_t elem_size;
    std::uint32_t flags;
};

class WeightArchive {
public:
    static WeightArchive load(const std::string& path);

    [[nodiscard]] const std::vector<TensorRecordMeta>& records() const noexcept {
        return records_;
    }

    [[nodiscard]] const TensorRecordMeta* find(const std::string& name) const noexcept;
    [[nodiscard]] TensorViewF32 tensor_f32(const std::string& name) const;

private:
    std::vector<std::byte> bytes_;
    std::vector<TensorRecordMeta> records_;
};

}  // namespace dlinf

