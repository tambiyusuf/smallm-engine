//
// Created by tambiyusuf on 6.07.2026.
//
#include "smallm/runtime/sampler/greedy_sampler.h"

namespace smallm {

    uint32_t GreedySampler::sample(const std::vector<float>& logits) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < logits.size(); ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

} // namespace smallm