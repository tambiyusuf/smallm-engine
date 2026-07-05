//
// Created by tambiyusuf on 6.07.2026.
//
#pragma once

#include <cstdint>
#include <vector>

namespace smallm {

    // picks the next token id from a logits vector. strategies (greedy, temperature,
    // top-k, top-p) implement this; the generator depends only on this interface.
    class Sampler {
    public:
        virtual ~Sampler() = default;
        virtual uint32_t sample(const std::vector<float>& logits) = 0;
    };

} // namespace smallm
