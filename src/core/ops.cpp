//
// Created by tambiyusuf on 4.07.2026.
//

#include "../../include/smallm/core/ops.h"

#include <cmath>
#include <algorithm>

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

} // namespace smallm::ops