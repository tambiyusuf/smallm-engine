//
// Created by tambiyusuf on 6.07.2026.
//
#pragma once

#include "smallm/runtime/sampler/common/sampler.h"

namespace smallm {

    // greedy: always pick the highest-logit token (deterministic)
    class GreedySampler : public Sampler {
    public:
        uint32_t sample(const std::vector<float>& logits) override;
    };

} // namespace smallm