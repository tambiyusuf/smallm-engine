//
// Created by tambiyusuf on 4.07.2026.
//
#include "../../include/smallm/core/tensor.h"

#include <cstring>
#include <stdexcept>

namespace smallm {

namespace {

// converts an IEEE half-precision (f16) value to float; scales are stored as f16
float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t bits;
    if (exp == 0) {
        // subnormal or zero: no implicit leading bit
        if (mant == 0) {
            bits = sign;                 // signed zero
        } else {
            // normalize the subnormal into a float
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // inf or nan
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        // normal number: rebias exponent from 15 to 127
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// locates a tensor's raw bytes inside the live mapping
const uint8_t* tensor_data_ptr(const GGUFModel& model, const GGUFTensorInfo& info) {
    return model.mapping + model.tensor_data_offset + info.offset;
}

// --- per-type dequantizers, each appending `count` floats to `out` ---

void dequant_f32(const uint8_t* src, uint64_t count, std::vector<float>& out) {
    // already float; just copy the raw bytes straight in
    const float* f = reinterpret_cast<const float*>(src);
    out.assign(f, f + count);
}

void dequant_q8_0(const uint8_t* src, uint64_t count, std::vector<float>& out) {
    const int block = 32;
    out.resize(count);

    uint64_t nblocks = count / block;
    for (uint64_t b = 0; b < nblocks; ++b) {
        // each block: 2-byte f16 scale, then 32 int8 quants
        uint16_t scale_bits;
        std::memcpy(&scale_bits, src, sizeof(scale_bits));
        float scale = f16_to_f32(scale_bits);

        const int8_t* q = reinterpret_cast<const int8_t*>(src + 2);
        for (int i = 0; i < block; ++i) {
            out[b * block + i] = scale * static_cast<float>(q[i]);
        }
        src += 2 + block;  // advance past this block (34 bytes)
    }
}

void dequant_q4_0(const uint8_t* src, uint64_t count, std::vector<float>& out) {
    const int block = 32;
    out.resize(count);

    uint64_t nblocks = count / block;
    for (uint64_t b = 0; b < nblocks; ++b) {
        // each block: 2-byte f16 scale, then 16 bytes packing 32 4-bit quants
        uint16_t scale_bits;
        std::memcpy(&scale_bits, src, sizeof(scale_bits));
        float scale = f16_to_f32(scale_bits);

        const uint8_t* q = src + 2;
        for (int i = 0; i < block / 2; ++i) {
            uint8_t packed = q[i];
            // low nibble is element i, high nibble is element i+16
            int lo = (packed & 0x0F) - 8;   // Q4_0 centers values around 0 (offset 8)
            int hi = (packed >> 4)   - 8;
            out[b * block + i]      = scale * static_cast<float>(lo);
            out[b * block + i + 16] = scale * static_cast<float>(hi);
        }
        src += 2 + block / 2;  // advance past this block (18 bytes)
    }
}

} // namespace

Tensor dequantize_tensor(const GGUFModel& model, const std::string& name) {
    // find the tensor info by name
    const GGUFTensorInfo* info = nullptr;
    for (const auto& t : model.tensors) {
        if (t.name == name) { info = &t; break; }
    }
    if (!info) throw std::runtime_error("tensor not found: " + name);

    Tensor out;
    out.dims = info->dims;
    uint64_t count = out.numel();

    const uint8_t* src = tensor_data_ptr(model, *info);

    switch (static_cast<TensorType>(info->type)) {
        case TensorType::F32:  dequant_f32(src, count, out.data);  break;
        case TensorType::Q8_0: dequant_q8_0(src, count, out.data); break;
        case TensorType::Q4_0: dequant_q4_0(src, count, out.data); break;
        default:
            throw std::runtime_error("unsupported tensor type: " + std::to_string(info->type));
    }
    return out;
}

    QuantizedTensor get_quantized_tensor(const GGUFModel& model, const std::string& name) {
    const GGUFTensorInfo* info = nullptr;
    for (const auto& t : model.tensors) {
        if (t.name == name) { info = &t; break; }
    }
    if (!info) throw std::runtime_error("tensor not found: " + name);

    QuantizedTensor qt;
    qt.data = model.mapping + model.tensor_data_offset + info->offset;
    qt.type = info->type;
    qt.dims = info->dims;
    return qt;
}

} // namespace smallm