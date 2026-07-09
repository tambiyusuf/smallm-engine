//
// Created by tambiyusuf on 9.07.2026.
//
#include "smallm/runtime/sampler/temperature_sampler.h"

#include <cmath>
#include <vector>

namespace smallm {

    TemperatureSampler::TemperatureSampler(float temperature, uint64_t seed)
        : temp_(temperature), rng_(seed) {}

    uint32_t TemperatureSampler::sample(const std::vector<float>& logits) {
        const size_t n = logits.size();

        // scale by 1/T and softmax (numerically stable: subtract max first)
        float maxv = logits[0];
        for (size_t i = 1; i < n; ++i) maxv = std::max(maxv, logits[i]);

        std::vector<float> probs(n);
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            probs[i] = std::exp((logits[i] - maxv) / temp_);
            sum += probs[i];
        }
        for (size_t i = 0; i < n; ++i) probs[i] /= sum;

        // sample one token from the distribution
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        float cumulative = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            cumulative += probs[i];
            if (r <= cumulative) return static_cast<uint32_t>(i);
        }
        return static_cast<uint32_t>(n - 1);  // fallback for float rounding
    }

} // namespace smallm