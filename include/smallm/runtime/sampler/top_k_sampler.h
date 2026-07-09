//
// Created by tambiyusuf on 9.07.2026.
//
#pragma once

#include "smallm/runtime/sampler/common/sampler.h"

#include <random>

namespace smallm {

    // top-k sampling: keep only the k highest-logit tokens, apply temperature,
    // softmax over them, and sample.
    class TopKSampler : public Sampler {
    public:
        TopKSampler(uint32_t k, float temperature, uint64_t seed = 0);
        uint32_t sample(const std::vector<float>& logits) override;

    private:
        uint32_t k_;
        float temp_;
        std::mt19937 rng_;
    };

} // namespace smallm