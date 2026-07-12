//
// Created by tambiyusuf on 5.07.2026.
//
#pragma once

#include "smallm/config/common/config.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace smallm {

    // abstract base every architecture implements; the runtime talks only to this
    class Model {
    public:
        virtual ~Model() = default;

        // runs one forward step for a single token at position `pos`,
        // returning logits over the vocabulary (size = config.vocab_size)
        virtual std::vector<float> forward(uint32_t token_id, uint32_t pos) = 0;

        // --- KV cache management (generic across architectures) ---

        // drop all cached state; the next forward starts from position 0
        virtual void reset_cache() = 0;

        // keep only the first `n` cached positions valid, discard the rest
        virtual void truncate_cache(uint32_t n) = 0;

        // how many positions currently hold valid cached state
        uint32_t cached_length() const { return cached_length_; }


        const ModelConfig& config() const { return *config_; }

    protected:
        explicit Model(std::unique_ptr<ModelConfig> cfg) : config_(std::move(cfg)) {}
        std::unique_ptr<ModelConfig> config_;
        // number of valid cached positions; kept in sync by forward/reset/truncate
        uint32_t cached_length_ = 0;
    };

    // builds the "blk.<i>.<suffix>" tensor name used by every architecture
    inline std::string layer_tensor_name(uint32_t i, const std::string& suffix) {
        return "blk." + std::to_string(i) + "." + suffix;
    }
} // namespace smallm