//
// Created by tambiyusuf on 5.07.2026.
//
#pragma once

#include "smallm/config/common/config.h"

namespace smallm {

    // Qwen2 currently needs nothing beyond the shared core, so this only marks the
    // architecture as a distinct type; architecture-specific fields land here later
    struct Qwen2Config : ModelConfig {
        // (no extra fields yet — Qwen2 uses only the core config)
    };

    // fills a Qwen2Config from metadata using the "qwen2." key prefix
    std::unique_ptr<Qwen2Config> read_qwen2_config(const GGUFModel& model);

} // namespace smallm