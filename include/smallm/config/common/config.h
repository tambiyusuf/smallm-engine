//
// Created by tambiyusuf on 5.07.2026.
//
#pragma once

#include "smallm/core/gguf.h"

#include <cstdint>
#include <memory>
#include <string>

namespace smallm {

    // shared core every transformer architecture needs; derived configs extend this
    struct ModelConfig {
        std::string architecture;

        uint32_t n_layers = 0;
        uint32_t embd_dim = 0;
        uint32_t ffn_dim  = 0;
        uint32_t n_heads  = 0;
        uint32_t n_kv_heads = 0;
        uint32_t vocab_size = 0;
        uint32_t context_length = 0;

        float rope_freq_base = 10000.0f;
        float rms_eps = 1e-5f;

        uint32_t head_dim() const {
            return n_heads == 0 ? 0 : embd_dim / n_heads;
        }

        // polymorphic base: derived configs add fields and must delete cleanly
        // through a base pointer
        virtual ~ModelConfig() = default;
    };

    // reads a config from metadata, dispatching on general.architecture;
    // returns the right derived type as a base pointer
    std::unique_ptr<ModelConfig> read_config(const GGUFModel& model);

    // fills the shared core fields from metadata using the given architecture prefix
    // (e.g. "qwen2", "llama"). architecture-specific readers call this first, then
    // add whatever extra fields their architecture needs.
    void read_core_config(const GGUFModel& model, const std::string& prefix,
                          ModelConfig& cfg);
} // namespace smallm