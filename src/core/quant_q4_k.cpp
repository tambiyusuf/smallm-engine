#include "smallm/core/common/quant_common.h"

#include <cstring>

namespace smallm {

namespace {

    constexpr int QK_K = 256;
    constexpr int BLOCK_Q4_K = 144;

    void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
        if (j < 4) {
            *d = q[j] & 63;
            *m = q[j + 4] & 63;
        } else {
            *d = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }

} // namespace

void dequantize_row_q4_k(const uint8_t* blocks, uint32_t k, float* out) {
    const int nb = static_cast<int>(k / QK_K);
    for (int i = 0; i < nb; ++i) {
        const uint8_t* q = blocks + i * BLOCK_Q4_K + 4 + 12;
        const uint8_t* scales = blocks + i * BLOCK_Q4_K + 4;

        uint16_t d_bits;
        uint16_t min_bits;
        std::memcpy(&d_bits, blocks + i * BLOCK_Q4_K, 2);
        std::memcpy(&min_bits, blocks + i * BLOCK_Q4_K + 2, 2);
        const float d = f16_to_f32(d_bits);
        const float min = f16_to_f32(min_bits);

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc = 0;
            uint8_t m = 0;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = min * static_cast<float>(m);
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = min * static_cast<float>(m);
            for (int l = 0; l < 32; ++l) {
                *out++ = d1 * static_cast<float>(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *out++ = d2 * static_cast<float>(q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

} // namespace smallm
