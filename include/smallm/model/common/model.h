//
// Created by tambiyusuf on 5.07.2026.
//
#pragma once

#include "smallm/config/common/config.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace smallm {

    // abstract base every architecture implements; the runtime talks only to this
    class Model {
    public:
        virtual ~Model() = default;

        // runs one forward step for a single token at position `pos`,
        // returning logits over the vocabulary (size = config.vocab_size)
        virtual std::vector<float> forward(uint32_t token_id, uint32_t pos) = 0;

        const ModelConfig& config() const { return *config_; }

    protected:
        explicit Model(std::unique_ptr<ModelConfig> cfg) : config_(std::move(cfg)) {}
        std::unique_ptr<ModelConfig> config_;
    };

} // namespace smallm