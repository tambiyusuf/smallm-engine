//
// Created by tambiyusuf on 4.07.2026.
//
#include "smallm/gguf.h"
#include "smallm/tensor.h"
#include "smallm/ops.h"

#include <iostream>
#include <variant>
#include <algorithm>
#include <cmath>

// prints a metadata value; only the common scalar types are spelled out here
static void print_value(const smallm::GGUFValue& v) {
    std::visit([](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::string>) {
            std::cout << x;
        } else if constexpr (std::is_arithmetic_v<T>) {
            std::cout << x;
        } else {
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

        std::cout << "--- metadata ---\n";
        for (const auto& [key, val] : model.metadata) {
            std::cout << key << " = ";
            print_value(val);
            std::cout << "\n";
        }

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

        // ---- dequantize checks ----
        std::cout << "\n--- dequantize test: blk.0.attn_norm.weight ---\n";
        smallm::Tensor t = smallm::dequantize_tensor(model, "blk.0.attn_norm.weight");
        std::cout << "numel = " << t.numel() << "\n";
        std::cout << "first 8 values: ";
        for (size_t i = 0; i < 8 && i < t.data.size(); ++i) std::cout << t.data[i] << " ";
        std::cout << "\n";

        std::cout << "\n--- dequantize test: blk.0.attn_output.weight (Q4_0) ---\n";
        smallm::Tensor q = smallm::dequantize_tensor(model, "blk.0.attn_output.weight");
        std::cout << "numel = " << q.numel() << "\n";
        std::cout << "first 8 values: ";
        for (size_t i = 0; i < 8 && i < q.data.size(); ++i) std::cout << q.data[i] << " ";
        std::cout << "\n";
        {
            float mn = q.data[0], mx = q.data[0]; double sum = 0;
            for (float v : q.data) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
            std::cout << "min=" << mn << " max=" << mx << " mean=" << sum / q.data.size() << "\n";
        }

        // ---- operator checks ----
        std::cout << "\n--- matmul test ---\n";
        // W = [[1,2,3],[4,5,6]] (2 rows, 3 cols), x = [1,0,1]
        {
            float W[6] = {1, 2, 3, 4, 5, 6};
            float x[3] = {1, 0, 1};
            float y[2] = {0, 0};
            smallm::ops::matmul(W, x, 2, 3, y);
            std::cout << "y = [" << y[0] << ", " << y[1] << "]  (expected [4, 10])\n";
        }

        std::cout << "\n--- softmax test ---\n";
        {
            float s[3] = {1.0f, 2.0f, 3.0f};
            smallm::ops::softmax(s, 3);
            std::cout << "softmax = [" << s[0] << ", " << s[1] << ", " << s[2] << "]\n";
            std::cout << "  sum = " << (s[0] + s[1] + s[2]) << "  (expected ~1.0)\n";
        }

        std::cout << "\n--- silu test ---\n";
        {
            float a[3] = {-1.0f, 0.0f, 2.0f};
            smallm::ops::silu(a, 3);
            std::cout << "silu = [" << a[0] << ", " << a[1] << ", " << a[2] << "]"
                      << "  (expected ~[-0.269, 0, 1.762])\n";
        }

        std::cout << "\n--- rmsnorm test ---\n";
        {
            float xr[4] = {1.0f, 2.0f, 3.0f, 4.0f};
            float wr[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            float yr[4] = {0, 0, 0, 0};
            float eps = 1e-6f;
            smallm::ops::rmsnorm(xr, wr, 4, eps, yr);
            std::cout << "rmsnorm = [" << yr[0] << ", " << yr[1] << ", "
                      << yr[2] << ", " << yr[3] << "]\n";
            float rms = std::sqrt((yr[0]*yr[0] + yr[1]*yr[1] + yr[2]*yr[2] + yr[3]*yr[3]) / 4.0f);
            std::cout << "  output rms = " << rms << "  (expected ~1.0)\n";
        }

        std::cout << "\n--- rope test ---\n";
        {
            // one head, head_dim = 4, so pairs are (idx0, idx2) and (idx1, idx3)
            float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};

            // at pos = 0 the angle is 0, so rope must leave the vector unchanged
            float v0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
            smallm::ops::rope(v0, 1, 4, 0, 10000.0f);
            std::cout << "pos=0: [" << v0[0] << ", " << v0[1] << ", "
                      << v0[2] << ", " << v0[3] << "]  (expected unchanged [1,2,3,4])\n";

            // at pos = 5 values change, but each rotated pair keeps its length
            smallm::ops::rope(v, 1, 4, 5, 10000.0f);
            float len_before_pair0 = 1.0f*1.0f + 3.0f*3.0f;   // (idx0, idx2) originally
            float len_after_pair0  = v[0]*v[0] + v[2]*v[2];
            std::cout << "pos=5: [" << v[0] << ", " << v[1] << ", "
                      << v[2] << ", " << v[3] << "]\n";
            std::cout << "  pair0 length before=" << len_before_pair0
                      << " after=" << len_after_pair0 << "  (should match)\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}