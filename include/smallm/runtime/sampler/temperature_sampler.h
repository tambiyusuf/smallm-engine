//
// Created by tambiyusuf on 9.07.2026.
//

#pragma once
#include "smallm/runtime/sampler/common/sampler.h"
#include <random>

namespace smallm {

    // temperature sampling: scale logits by 1/T, softmax, then sample from the
    // distribution. lower T is sharper/safer, higher T is flatter/more diverse.
    class TemperatureSampler : public Sampler {
    public:
        explicit TemperatureSampler(float temperature, uint64_t seed=0);
        uint32_t sample(const std::vector<float>& logits) override;

    private:
        float temp_;
        std::mt19937 rng_;
    };
}