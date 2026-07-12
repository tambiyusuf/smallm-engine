//
// Created by tambiyusuf on 4.07.2026.
//
#pragma once

#include "gguf.h"

#include <cstdint>
#include <vector>

namespace smallm {

    // GGUF tensor type ids we currently understand
    enum class TensorType : uint32_t {
        F32   = 0,
        F16   = 1,
        Q4_0  = 2,
        Q8_0  = 8,
    };

    // a tensor's weights fully expanded to float, plus its shape
    struct Tensor {
        std::vector<uint64_t> dims;   // shape, same order as in the file
        std::vector<float> data;      // dequantized values, row-major

        // total element count across all dimensions
        uint64_t numel() const {
            uint64_t n = 1;
            for (uint64_t d : dims) n *= d;
            return n;
        }
    };

    // a tensor left in its quantized form: a zero-copy view into the mmap'd weights,
    // dequantized on the fly during matmul instead of up front.
    struct QuantizedTensor {
        const uint8_t* data = nullptr;   // raw quantized bytes in the mapping
        uint32_t type = 0;               // GGUF type id (Q4_0, Q8_0, F32...)
        std::vector<uint64_t> dims;      // shape, same order as in the file

        uint64_t numel() const {
            uint64_t n = 1;
            for (uint64_t d : dims) n *= d;
            return n;
        }
    };

    // builds a QuantizedTensor pointing at a tensor's raw bytes, without dequantizing
    QuantizedTensor get_quantized_tensor(const GGUFModel& model, const std::string& name);
    // finds a tensor by name in the model, dequantizes it to float, and returns it;
    // throws if the name is missing or the type isn't supported yet
    Tensor dequantize_tensor(const GGUFModel& model, const std::string& name);

} // namespace smallm