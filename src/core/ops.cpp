//
// Created by tambiyusuf on 4.07.2026.
//

#include "../../include/smallm/core/ops.h"
#include "smallm/core/common/quant_common.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <immintrin.h>   // AVX2 intrinsics
#include <cstring>
namespace smallm::ops {

    // naive matrix-vector product; correctness first, speed later
    void matmul(const float* W, const float* x,
                uint32_t rows, uint32_t cols,
                float* y) {
        for (uint32_t r = 0; r < rows; ++r) {
            float acc = 0.0f;
            const float* row = W + static_cast<uint64_t>(r) * cols;
            for (uint32_t c = 0; c < cols; ++c) {
                acc += row[c] * x[c];
            }
            y[r] = acc;
        }
    }

    // RMS normalization: scale x by its root-mean-square, then apply per-element weight
    void rmsnorm(const float* x, const float* weight,
                 uint32_t size, float eps,
                 float* y) {
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < size; ++i) sum_sq += x[i] * x[i];
        float mean_sq = sum_sq / static_cast<float>(size);

        float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
        for (uint32_t i = 0; i < size; ++i) {
            y[i] = x[i] * inv_rms * weight[i];
        }
    }

    // numerically stable softmax over `size` elements, in place
    void softmax(float* x, uint32_t size) {
        if (size == 0) return;

        float maxv = x[0];
        for (uint32_t i = 1; i < size; ++i) maxv = std::max(maxv, x[i]);

        float sum = 0.0f;
        for (uint32_t i = 0; i < size; ++i) {
            x[i] = std::exp(x[i] - maxv);
            sum += x[i];
        }

        float inv = 1.0f / sum;
        for (uint32_t i = 0; i < size; ++i) x[i] *= inv;
    }

    // SiLU (swish): x * sigmoid(x), in place
    void silu(float* x, uint32_t size) {
        for (uint32_t i = 0; i < size; ++i) {
            float v = x[i];
            x[i] = v / (1.0f + std::exp(-v));
        }
    }

    // applies rotary position embedding in place, per head, on dimension pairs
    void rope(float* x, uint32_t n_heads, uint32_t head_dim,
              uint32_t pos, float freq_base) {
        // rotate each head independently
        for (uint32_t h = 0; h < n_heads; ++h) {
            float* head = x + static_cast<uint64_t>(h) * head_dim;

            // walk the head in pairs: (0,1), (2,3), ...
            for (uint32_t i = 0; i < head_dim / 2; ++i) {
                // frequency for this pair; earlier pairs rotate faster
                float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
                float freq = 1.0f / std::pow(freq_base, exponent);
                float theta = static_cast<float>(pos) * freq;

                float cos_t = std::cos(theta);
                float sin_t = std::sin(theta);

                // the pair we rotate: element i and element i + head_dim/2
                float x0 = head[i];
                float x1 = head[i + head_dim / 2];

                head[i]                = x0 * cos_t - x1 * sin_t;
                head[i + head_dim / 2] = x0 * sin_t + x1 * cos_t;
            }
        }
    }

    // dequantizes and multiplies one row of a Q4_0/Q8_0 weight with x, returning the dot.
// blocks of 32 values: Q4_0 = 2-byte scale + 16 packed bytes, Q8_0 = 2-byte scale + 32 int8.

void matmul_quantized(const uint8_t* W, uint32_t type,
                      const float* x, uint32_t rows, uint32_t cols,
                      float* y) {
    const int block = 32;
    const uint32_t nblocks = cols / block;

    if (type == 8) {  // Q8_0: 2-byte scale + 32 int8 per block
        const uint32_t row_bytes = nblocks * (2 + block);  // 34 bytes/block

        #pragma omp parallel for
        for (uint32_t r = 0; r < rows; ++r) {
            const uint8_t* row = W + static_cast<uint64_t>(r) * row_bytes;
            __m256 acc = _mm256_setzero_ps();

            for (uint32_t b = 0; b < nblocks; ++b) {
                uint16_t sb;
                std::memcpy(&sb, row, 2);
                __m256 vscale = _mm256_set1_ps(f16_to_f32(sb));

                const int8_t* q = reinterpret_cast<const int8_t*>(row + 2);
                const float* xb = x + b * block;

                // 32 signed int8 values, 8 lanes at a time, all in registers
                for (int i = 0; i < block; i += 8) {
                    __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + i));
                    __m256i wide  = _mm256_cvtepi8_epi32(bytes);   // signed widen
                    __m256  wf    = _mm256_cvtepi32_ps(wide);
                    __m256  vx    = _mm256_loadu_ps(xb + i);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(wf, vscale), vx, acc);
                }
                row += 2 + block;
            }

            // horizontal sum of the 8 lanes
            __m128 lo128 = _mm256_castps256_ps128(acc);
            __m128 hi128 = _mm256_extractf128_ps(acc, 1);
            lo128 = _mm_add_ps(lo128, hi128);
            lo128 = _mm_hadd_ps(lo128, lo128);
            lo128 = _mm_hadd_ps(lo128, lo128);
            y[r] = _mm_cvtss_f32(lo128);
        }

    } else if (type == 2) {  // Q4_0: 2-byte scale + 16 packed bytes per block
        const uint32_t row_bytes = nblocks * (2 + block / 2);  // 18 bytes/block

        const __m256i low_mask = _mm256_set1_epi32(0x0F);
        const __m256i eight    = _mm256_set1_epi32(8);

        #pragma omp parallel for
        for (uint32_t r = 0; r < rows; ++r) {
            const uint8_t* row = W + static_cast<uint64_t>(r) * row_bytes;
            __m256 acc = _mm256_setzero_ps();

            for (uint32_t b = 0; b < nblocks; ++b) {
                uint16_t sb;
                std::memcpy(&sb, row, 2);
                __m256 vscale = _mm256_set1_ps(f16_to_f32(sb));

                const uint8_t* q = row + 2;
                const float* xb = x + b * block;

                // 8 packed bytes -> 8 low nibbles (i..i+7) and 8 high nibbles (i+16..i+23)
                for (int i = 0; i < 16; i += 8) {
                    __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + i));
                    __m256i wide  = _mm256_cvtepu8_epi32(bytes);   // unsigned widen

                    // low nibbles: (b & 0x0F) - 8
                    __m256i lo_i = _mm256_sub_epi32(_mm256_and_si256(wide, low_mask), eight);
                    __m256  lo_f = _mm256_cvtepi32_ps(lo_i);
                    __m256  x_lo = _mm256_loadu_ps(xb + i);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(lo_f, vscale), x_lo, acc);

                    // high nibbles: (b >> 4) - 8
                    __m256i hi_i = _mm256_sub_epi32(_mm256_srli_epi32(wide, 4), eight);
                    __m256  hi_f = _mm256_cvtepi32_ps(hi_i);
                    __m256  x_hi = _mm256_loadu_ps(xb + 16 + i);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(hi_f, vscale), x_hi, acc);
                }
                row += 2 + block / 2;
            }

            __m128 lo128 = _mm256_castps256_ps128(acc);
            __m128 hi128 = _mm256_extractf128_ps(acc, 1);
            lo128 = _mm_add_ps(lo128, hi128);
            lo128 = _mm_hadd_ps(lo128, lo128);
            lo128 = _mm_hadd_ps(lo128, lo128);
            y[r] = _mm_cvtss_f32(lo128);
        }
    } else if (type == 14) {  // Q6_K, AVX2
        const int QK_K = 256;
        const int block_bytes = 210;
        const uint32_t nsb = cols / QK_K;
        const uint32_t row_bytes = nsb * block_bytes;

        const __m256i mask_0f = _mm256_set1_epi32(0x0F);
        const __m256i mask_03 = _mm256_set1_epi32(0x03);
        const __m256i c32     = _mm256_set1_epi32(32);

        #pragma omp parallel for
        for (uint32_t r = 0; r < rows; ++r) {
            const uint8_t* row = W + static_cast<uint64_t>(r) * row_bytes;
            __m256 acc = _mm256_setzero_ps();

            for (uint32_t sb = 0; sb < nsb; ++sb) {
                const uint8_t* ql     = row;
                const uint8_t* qh     = row + 128;
                const int8_t*  scales = reinterpret_cast<const int8_t*>(row + 192);

                uint16_t d_bits;
                std::memcpy(&d_bits, row + 208, 2);
                __m256 vd = _mm256_set1_ps(f16_to_f32(d_bits));

                const float* xb = x + sb * QK_K;

                for (int half = 0; half < 2; ++half) {
                    const uint8_t* ql_h = ql + half * 64;
                    const uint8_t* qh_h = qh + half * 32;
                    const int8_t*  sc_h = scales + half * 8;
                    const float*   x_h  = xb + half * 128;

                    // process 8 values of l at a time (l = 0..31)
                    for (int l = 0; l < 32; l += 8) {
                        // sub-block scales: all 8 lanes in this chunk share l/16
                        int si = l / 16;
                        __m256 s0 = _mm256_set1_ps(static_cast<float>(sc_h[si]));
                        __m256 s1 = _mm256_set1_ps(static_cast<float>(sc_h[2 + si]));
                        __m256 s2 = _mm256_set1_ps(static_cast<float>(sc_h[4 + si]));
                        __m256 s3 = _mm256_set1_ps(static_cast<float>(sc_h[6 + si]));

                        // widen the packed bytes into int32 lanes
                        __m256i vql_lo = _mm256_cvtepu8_epi32(
                            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql_h + l)));
                        __m256i vql_hi = _mm256_cvtepu8_epi32(
                            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql_h + l + 32)));
                        __m256i vqh = _mm256_cvtepu8_epi32(
                            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh_h + l)));

                        // q0 = (ql_lo & 0x0F) | ((qh >> 0 & 3) << 4) - 32
                        __m256i q0 = _mm256_sub_epi32(
                            _mm256_or_si256(
                                _mm256_and_si256(vql_lo, mask_0f),
                                _mm256_slli_epi32(_mm256_and_si256(vqh, mask_03), 4)),
                            c32);

                        // q1 = (ql_hi & 0x0F) | ((qh >> 2 & 3) << 4) - 32
                        __m256i q1 = _mm256_sub_epi32(
                            _mm256_or_si256(
                                _mm256_and_si256(vql_hi, mask_0f),
                                _mm256_slli_epi32(
                                    _mm256_and_si256(_mm256_srli_epi32(vqh, 2), mask_03), 4)),
                            c32);

                        // q2 = (ql_lo >> 4) | ((qh >> 4 & 3) << 4) - 32
                        __m256i q2 = _mm256_sub_epi32(
                            _mm256_or_si256(
                                _mm256_srli_epi32(vql_lo, 4),
                                _mm256_slli_epi32(
                                    _mm256_and_si256(_mm256_srli_epi32(vqh, 4), mask_03), 4)),
                            c32);

                        // q3 = (ql_hi >> 4) | ((qh >> 6 & 3) << 4) - 32
                        __m256i q3 = _mm256_sub_epi32(
                            _mm256_or_si256(
                                _mm256_srli_epi32(vql_hi, 4),
                                _mm256_slli_epi32(
                                    _mm256_and_si256(_mm256_srli_epi32(vqh, 6), mask_03), 4)),
                            c32);

                        // convert to float, apply d * sub-scale, multiply with x, accumulate
                        __m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q0), _mm256_mul_ps(vd, s0));
                        __m256 f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q1), _mm256_mul_ps(vd, s1));
                        __m256 f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(q2), _mm256_mul_ps(vd, s2));
                        __m256 f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(q3), _mm256_mul_ps(vd, s3));

                        acc = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x_h + l),      acc);
                        acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x_h + l + 32), acc);
                        acc = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x_h + l + 64), acc);
                        acc = _mm256_fmadd_ps(f3, _mm256_loadu_ps(x_h + l + 96), acc);
                    }
                }
                row += block_bytes;
            }

            // horizontal sum
            __m128 lo128 = _mm256_castps256_ps128(acc);
            __m128 hi128 = _mm256_extractf128_ps(acc, 1);
            lo128 = _mm_add_ps(lo128, hi128);
            lo128 = _mm_hadd_ps(lo128, lo128);
            lo128 = _mm_hadd_ps(lo128, lo128);
            y[r] = _mm_cvtss_f32(lo128);
        }
    }
        else {  // F32 fallback: plain float weights
        const float* Wf = reinterpret_cast<const float*>(W);

        #pragma omp parallel for
        for (uint32_t r = 0; r < rows; ++r) {
            const float* wr = Wf + static_cast<uint64_t>(r) * cols;
            float acc = 0.0f;
            for (uint32_t c = 0; c < cols; ++c) acc += wr[c] * x[c];
            y[r] = acc;
        }
    }
}


    // AVX2 dot product: 8 lanes at a time, scalar tail for the remainder
    float dot(const float* a, const float* b, uint32_t n) {
        __m256 acc = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        // horizontal sum of the 8 lanes
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float sum = _mm_cvtss_f32(lo);

        // leftover elements
        for (; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }

    // AVX2 y += scale * v
    void axpy(float* y, const float* v, float scale, uint32_t n) {
        __m256 vscale = _mm256_set1_ps(scale);
        uint32_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 vy = _mm256_loadu_ps(y + i);
            __m256 vv = _mm256_loadu_ps(v + i);
            vy = _mm256_fmadd_ps(vv, vscale, vy);
            _mm256_storeu_ps(y + i, vy);
        }
        for (; i < n; ++i) y[i] += scale * v[i];
    }

    void mul(float* y, const float* v, uint32_t n) {
        uint32_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 vy = _mm256_loadu_ps(y + i);
            __m256 vv = _mm256_loadu_ps(v + i);
            _mm256_storeu_ps(y + i, _mm256_mul_ps(vy, vv));
        }
        for (; i < n; ++i) y[i] *= v[i];
    }

} // namespace smallm::ops