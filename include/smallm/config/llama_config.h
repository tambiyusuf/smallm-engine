//
// Created by tambiyusuf on 12.07.2026.
//
#pragma once

#include "smallm/config/common/config.h"

namespace smallm {

    // Llama needs nothing beyond the shared core config
    struct LlamaConfig : ModelConfig {
        // (no extra fields — Llama uses only the core config)
    };

    // fills a LlamaConfig from metadata using the "llama." key prefix
    std::unique_ptr<LlamaConfig> read_llama_config(const GGUFModel& model);

} // namespace smallm