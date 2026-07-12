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
void dequant_q6_k(const uint8_t* src, uint64_t count, std::vector<float>& out) {
    const int QK_K = 256;          // values per super-block
    const int block_bytes = 210;   // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
    out.resize(count);

    uint64_t nblocks = count / QK_K;
    for (uint64_t b = 0; b < nblocks; ++b) {
        const uint8_t* ql     = src;
        const uint8_t* qh     = src + 128;
        const int8_t*  scales = reinterpret_cast<const int8_t*>(src + 192);

        uint16_t d_bits;
        std::memcpy(&d_bits, src + 208, 2);
        float d = f16_to_f32(d_bits);

        float* dst = out.data() + b * QK_K;

        // the 256 values are packed as two halves of 128, each in groups of 32.
        // for each l in 0..31, four values are extracted from ql[l], ql[l+32] and qh[l].
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql_h = ql + half * 64;
            const uint8_t* qh_h = qh + half * 32;
            const int8_t*  sc_h = scales + half * 8;
            float* dst_h = dst + half * 128;

            for (int l = 0; l < 32; ++l) {
                uint8_t h = qh_h[l];

                // 6-bit values: low nibble from ql, high 2 bits from qh, centered at 32
                int q0 = static_cast<int>((ql_h[l]      & 0x0F) | (((h >> 0) & 3) << 4)) - 32;
                int q1 = static_cast<int>((ql_h[l + 32] & 0x0F) | (((h >> 2) & 3) << 4)) - 32;
                int q2 = static_cast<int>((ql_h[l]      >>   4) | (((h >> 4) & 3) << 4)) - 32;
                int q3 = static_cast<int>((ql_h[l + 32] >>   4) | (((h >> 6) & 3) << 4)) - 32;

                // each group of 16 values shares a sub-block scale
                dst_h[l]      = d * sc_h[l / 16]       * static_cast<float>(q0);
                dst_h[l + 32] = d * sc_h[2 + l / 16]   * static_cast<float>(q1);
                dst_h[l + 64] = d * sc_h[4 + l / 16]   * static_cast<float>(q2);
                dst_h[l + 96] = d * sc_h[6 + l / 16]   * static_cast<float>(q3);
            }
        }
        src += block_bytes;
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
        case TensorType::Q6_K: dequant_q6_k(src, count, out.data); break;
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