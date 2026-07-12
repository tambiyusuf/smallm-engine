//
// Created by tambiyusuf on 5.07.2026.
//
#include "smallm/config/qwen2_config.h"

namespace smallm {

    std::unique_ptr<Qwen2Config> read_qwen2_config(const GGUFModel& model) {
        auto cfg = std::make_unique<Qwen2Config>();
        read_core_config(model, "qwen2", *cfg);
        // (no Qwen2-specific fields yet; they would go here)
        return cfg;
    }

} // namespace smallm