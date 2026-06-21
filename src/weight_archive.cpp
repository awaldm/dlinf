#include "dlinf/weight_archive.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace eigen_learn {
namespace {

constexpr char kMagic[8] = {'E', 'L', 'W', 'T', '0', '0', '0', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kEndianTag = 0x01020304;
constexpr std::uint32_t kHeaderBytes = 64;
constexpr std::uint32_t kRecordBytes = 224;
constexpr std::uint32_t kFlagCContiguous = 1;

#pragma pack(push, 1)
struct FileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t endian_tag;
    std::uint32_t header_bytes;
    std::uint32_t record_bytes;
    std::uint64_t tensor_count;
    std::uint64_t records_offset;
    std::uint64_t data_offset;
    std::uint64_t file_bytes;
    std::uint64_t reserved;
};

struct TensorRecord {
    char name[128];
    std::uint32_t dtype;
    std::uint32_t rank;
    std::uint64_t offset;
    std::uint64_t nbytes;
    std::uint32_t elem_size;
    std::uint32_t flags;
    std::uint64_t shape[8];
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == kHeaderBytes);
static_assert(sizeof(TensorRecord) == kRecordBytes);

bool host_is_little_endian() {
    const std::uint16_t value = 1;
    return *reinterpret_cast<const unsigned char*>(&value) == 1;
}

std::string read_name(const char (&name)[128]) {
    const auto* begin = name;
    const auto* end = std::find(begin, begin + 128, '\0');
    return std::string(begin, end);
}

void require_range(
    std::uint64_t offset,
    std::uint64_t nbytes,
    std::uint64_t file_bytes,
    const std::string& what) {
    if (offset > file_bytes || nbytes > file_bytes - offset) {
        throw std::runtime_error("ELW archive has out-of-range " + what);
    }
}

}  // namespace

WeightArchive WeightArchive::load(const std::string& path) {
    if (!host_is_little_endian()) {
        throw std::runtime_error("ELW loader currently requires a little-endian host");
    }

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open weight archive: " + path);
    }

    const auto end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("failed to determine weight archive size: " + path);
    }
    const auto file_size = static_cast<std::uint64_t>(end);
    if (file_size < sizeof(FileHeader)) {
        throw std::runtime_error("ELW archive is smaller than the header");
    }
    if (file_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("ELW archive is too large for this process");
    }

    WeightArchive archive;
    archive.bytes_.resize(static_cast<std::size_t>(file_size));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(archive.bytes_.data()), static_cast<std::streamsize>(file_size));
    if (!input) {
        throw std::runtime_error("failed to read complete weight archive: " + path);
    }

    const auto* header = reinterpret_cast<const FileHeader*>(archive.bytes_.data());
    if (std::memcmp(header->magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("invalid ELW magic");
    }
    if (header->version != kVersion || header->endian_tag != kEndianTag) {
        throw std::runtime_error("unsupported ELW version or endian tag");
    }
    if (header->header_bytes != kHeaderBytes || header->record_bytes != kRecordBytes) {
        throw std::runtime_error("unsupported ELW record layout");
    }
    if (header->file_bytes != file_size) {
        throw std::runtime_error("ELW file size does not match header");
    }
    if (header->reserved != 0) {
        throw std::runtime_error("ELW reserved header field must be zero");
    }

    const std::uint64_t records_bytes = header->tensor_count * kRecordBytes;
    require_range(header->records_offset, records_bytes, file_size, "record table");
    require_range(header->data_offset, 0, file_size, "data offset");

    archive.records_.reserve(static_cast<std::size_t>(header->tensor_count));
    const auto* records_begin = archive.bytes_.data() + header->records_offset;
    for (std::uint64_t i = 0; i < header->tensor_count; ++i) {
        const auto* raw = reinterpret_cast<const TensorRecord*>(records_begin + i * kRecordBytes);
        if (raw->rank > 8) {
            throw std::runtime_error("ELW tensor rank exceeds 8");
        }
        require_range(raw->offset, raw->nbytes, file_size, "tensor payload");

        TensorRecordMeta meta;
        meta.name = read_name(raw->name);
        meta.dtype = static_cast<DType>(raw->dtype);
        meta.offset = raw->offset;
        meta.nbytes = raw->nbytes;
        meta.elem_size = raw->elem_size;
        meta.flags = raw->flags;
        meta.shape.assign(raw->shape, raw->shape + raw->rank);
        archive.records_.push_back(std::move(meta));
    }

    return archive;
}

const TensorRecordMeta* WeightArchive::find(const std::string& name) const noexcept {
    const auto iter = std::find_if(
        records_.begin(),
        records_.end(),
        [&name](const TensorRecordMeta& record) { return record.name == name; });
    return iter == records_.end() ? nullptr : &*iter;
}

TensorViewF32 WeightArchive::tensor_f32(const std::string& name) const {
    const TensorRecordMeta* record = find(name);
    if (record == nullptr) {
        throw std::runtime_error("tensor not found in ELW archive: " + name);
    }
    if (record->dtype != DType::Float32 || record->elem_size != sizeof(float)) {
        throw std::runtime_error("tensor is not float32: " + name);
    }
    if ((record->flags & kFlagCContiguous) == 0) {
        throw std::runtime_error("tensor is not C-contiguous: " + name);
    }
    if (record->nbytes % sizeof(float) != 0) {
        throw std::runtime_error("float32 tensor byte count is not divisible by 4: " + name);
    }

    const auto* data = reinterpret_cast<const float*>(bytes_.data() + record->offset);
    return TensorViewF32(data, record->shape);
}

}  // namespace eigen_learn
