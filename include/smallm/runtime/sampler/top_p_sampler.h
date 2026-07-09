//
// Created by tambiyusuf on 9.07.2026.
//
#pragma once

#include "smallm/runtime/sampler/common/sampler.h"

#include <random>

namespace smallm {

    // top-p (nucleus) sampling: keep the smallest set of top tokens whose cumulative
    // probability reaches p, apply temperature, then sample from that set.
    class TopPSampler : public Sampler {
    public:
        TopPSampler(float p, float temperature, uint64_t seed = 0);
        uint32_t sample(const std::vector<float>& logits) override;

    private:
        float p_;
        float temp_;
        std::mt19937 rng_;
    };

} // namespace smallm