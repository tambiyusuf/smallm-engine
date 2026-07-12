//
// Created by tambiyusuf on 13.07.2026.
//

#pragma once

#include <cstdint>
#include <cstring>

namespace smallm {

    // converts an IEEE half-precision (f16) value to float.
    // quantized formats store their block scales as f16, so both the tensor
    // dequantizer and the quantized matmul need this.
    inline float f16_to_f32(uint16_t h) {
        uint32_t sign = (h & 0x8000u) << 16;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;                 // signed zero
            } else {
                // normalize the subnormal
                exp = 127 - 15 + 1;
                while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
                mant &= 0x3FF;
                bits = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 0x1F) {
            bits = sign | 0x7F800000u | (mant << 13);   // inf / nan
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }

        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

} // namespace smallm