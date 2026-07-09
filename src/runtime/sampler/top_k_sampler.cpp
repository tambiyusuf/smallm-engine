//
// Created by tambiyusuf on 9.07.2026.
//
#include "smallm/runtime/sampler/top_k_sampler.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace smallm {

    TopKSampler::TopKSampler(uint32_t k, float temperature, uint64_t seed)
        : k_(k), temp_(temperature), rng_(seed) {}

    uint32_t TopKSampler::sample(const std::vector<float>& logits) {
        const size_t n = logits.size();

        // pair each logit with its index so we can rank and still know the token id
        std::vector<std::pair<float, uint32_t>> scored(n);
        for (size_t i = 0; i < n; ++i) scored[i] = {logits[i], static_cast<uint32_t>(i)};

        // partial sort: bring the k highest logits to the front
        uint32_t k = std::min<uint32_t>(k_, static_cast<uint32_t>(n));
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        // softmax over just the top-k (with temperature)
        float maxv = scored[0].first;
        std::vector<float> probs(k);
        float sum = 0.0f;
        for (uint32_t i = 0; i < k; ++i) {
            probs[i] = std::exp((scored[i].first - maxv) / temp_);
            sum += probs[i];
        }
        for (uint32_t i = 0; i < k; ++i) probs[i] /= sum;

        // sample within the top-k, then map back to the real token id
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        float cumulative = 0.0f;
        for (uint32_t i = 0; i < k; ++i) {
            cumulative += probs[i];
            if (r <= cumulative) return scored[i].second;
        }
        return scored[k - 1].second;
    }

} // namespace smallm