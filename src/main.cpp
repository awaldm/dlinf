#include "dlinf/activation.hpp"
#include "dlinf/weight_archive.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

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

void run_relu_smoke_test() {
    std::vector<float> values = {-2.0f, -0.5f, 0.0f, 3.0f, 7.5f};
    dlinf::relu_inplace(values.data(), values.size());

    std::cout << "ReLU smoke test:";
    for (float value : values) {
        std::cout << " " << value;
    }
    std::cout << "\n";
}

void print_tensor_preview(const dlinf::TensorViewF32& t, const char* name) {
    std::cout << name << " shape ";
    print_shape(t.shape());
    std::cout << " elements=" << t.size() << "\n";

    if (t.size() == 0) return;

    float min_val = t.data()[0], max_val = t.data()[0];
    double sum = 0.0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        float v = t.data()[i];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }
    std::cout << "  min=" << min_val << " max=" << max_val
              << " mean=" << (sum / t.size()) << "\n";

    // number of elements to print
    const std::size_t n = std::min<std::size_t>(5, t.size());

    // first n elements
    std::cout << "  first " << n << ": ";
    for (std::size_t i = 0; i < n; ++i) std::cout << t.data()[i] << " ";

    // last n elements
    if (t.size() > n) {
        std::cout << "\n  last  " << n << ": ";
        for (std::size_t i = t.size() - n; i < t.size(); ++i) std::cout << t.data()[i] << " ";
    }
    std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run_relu_smoke_test();

        if (argc < 2) {
            std::cout << "Pass a .elw archive path to inspect exported weights.\n";
            return 0;
        }
        // Load the archive
        const std::string archive_path = argv[1];
        const auto archive = dlinf::WeightArchive::load(archive_path);
        std::cout << "Loaded archive: " << archive_path << "\n";
        std::cout << "Tensor records: " << archive.records().size() << "\n";

        // Assign them vars
        auto weight = archive.tensor_f32("fc.weight");
        auto bias = archive.tensor_f32("fc.bias");

        std::cout << "\n-----------------------------------\n";

        print_tensor_preview(weight, "fc.weight"); // single tensor details

        std::cout << "\n-----------------------------------\n";

        //auto output = dlinf::linear(input, weight, bias);

        std::cout << "\n-----------------------------------\n";

        const auto preview_count = std::min<std::size_t>(archive.records().size(), 8);
        for (std::size_t i = 0; i < preview_count; ++i) {
            const auto& record = archive.records()[i];
            std::cout << "  " << record.name << " ";
            print_shape(record.shape);
            std::cout << " bytes=" << record.nbytes << "\n";
        }

        if (const auto* fc = archive.find("fc.weight")) {
            std::cout << "fc.weight shape ";
            print_shape(fc->shape);
            std::cout << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}

