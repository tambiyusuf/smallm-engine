//
// Created by tambiyusuf on 4.07.2026.
//
#include "smallm/gguf.h"

#include <iostream>
#include <variant>

#include "smallm/tensor.h"

// prints a metadata value; only the common scalar types are spelled out here
static void print_value(const smallm::GGUFValue& v) {
    std::visit([](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::string>) {
            std::cout << x;
        } else if constexpr (std::is_arithmetic_v<T>) {
            std::cout << x;
        } else {
            // arrays and everything else just show their kind, not the contents
            std::cout << "[array/complex]";
        }
    }, v.data);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: smallm <model.gguf>\n";
        return 1;
    }

    try {
        smallm::GGUFModel model = smallm::load_gguf(argv[1]);

        std::cout << "GGUF version:   " << model.version << "\n";
        std::cout << "tensor count:   " << model.tensor_count << "\n";
        std::cout << "metadata count: " << model.metadata_count << "\n";
        std::cout << "data offset:    " << model.tensor_data_offset << "\n\n";

        // dump metadata keys with their scalar values
        std::cout << "--- metadata ---\n";
        for (const auto& [key, val] : model.metadata) {
            std::cout << key << " = ";
            print_value(val);
            std::cout << "\n";
        }

        // list the first few tensors so we can eyeball shapes and types
        std::cout << "\n--- tensors (first 10) ---\n";
        for (size_t i = 0; i < model.tensors.size() && i < 10; ++i) {
            const auto& t = model.tensors[i];
            std::cout << t.name << "  type=" << t.type << "  dims=[";
            for (size_t d = 0; d < t.dims.size(); ++d) {
                std::cout << t.dims[d];
                if (d + 1 < t.dims.size()) std::cout << ", ";
            }
            std::cout << "]  offset=" << t.offset << "\n";
        }

        // dequantize one small tensor and print its first values as a sanity check
        std::cout << "\n--- dequantize test: blk.0.attn_norm.weight ---\n";
        smallm::Tensor t = smallm::dequantize_tensor(model, "blk.0.attn_norm.weight");
        std::cout << "numel = " << t.numel() << "\n";
        std::cout << "first 8 values: ";
        for (size_t i = 0; i < 8 && i < t.data.size(); ++i) {
            std::cout << t.data[i] << " ";
        }
        std::cout << "\n";

        // test 2: a Q4_0 weight matrix — checks the nibble unpacking path
        std::cout << "\n--- dequantize test: blk.0.attn_output.weight (Q4_0) ---\n";
        smallm::Tensor q = smallm::dequantize_tensor(model, "blk.0.attn_output.weight");
        std::cout << "numel = " << q.numel() << "\n";
        std::cout << "first 8 values: ";
        for (size_t i = 0; i < 8 && i < q.data.size(); ++i) std::cout << q.data[i] << " ";
        std::cout << "\n";

        // simple stats to sanity-check the distribution
        float mn = q.data[0], mx = q.data[0]; double sum = 0;
        for (float v : q.data) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
        std::cout << "min=" << mn << " max=" << mx << " mean=" << sum / q.data.size() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}