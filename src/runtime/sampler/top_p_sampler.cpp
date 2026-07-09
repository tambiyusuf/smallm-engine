//
// Created by tambiyusuf on 9.07.2026.
//
#include "smallm/runtime/sampler/top_p_sampler.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace smallm {

    TopPSampler::TopPSampler(float p, float temperature, uint64_t seed)
        : p_(p), temp_(temperature), rng_(seed) {}

    uint32_t TopPSampler::sample(const std::vector<float>& logits) {
        const size_t n = logits.size();

        std::vector<std::pair<float, uint32_t>> scored(n);
        for (size_t i = 0; i < n; ++i) scored[i] = {logits[i], static_cast<uint32_t>(i)};

        // full sort by logit, descending (nucleus needs the ranked distribution)
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // softmax over all (with temperature), in ranked order
        float maxv = scored[0].first;
        std::vector<float> probs(n);
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            probs[i] = std::exp((scored[i].first - maxv) / temp_);
            sum += probs[i];
        }
        for (size_t i = 0; i < n; ++i) probs[i] /= sum;

        // keep the smallest prefix whose cumulative probability reaches p
        float cumulative = 0.0f;
        size_t cutoff = n;
        for (size_t i = 0; i < n; ++i) {
            cumulative += probs[i];
            if (cumulative >= p_) { cutoff = i + 1; break; }
        }

        // renormalize over the kept nucleus and sample from it
        float nucleus_sum = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) nucleus_sum += probs[i];

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_) * nucleus_sum;   // scale into the nucleus mass
        float acc = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) {
            acc += probs[i];
            if (r <= acc) return scored[i].second;
        }
        return scored[cutoff - 1].second;
    }

} // namespace smallm