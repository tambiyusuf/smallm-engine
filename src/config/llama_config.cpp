//
// Created by tambiyusuf on 12.07.2026.
//

#include "smallm/config/llama_config.h"

namespace smallm {

    std::unique_ptr<LlamaConfig> read_llama_config(const GGUFModel& model) {
        auto cfg = std::make_unique<LlamaConfig>();
        read_core_config(model, "llama", *cfg);
        // (no Llama-specific fields yet; they would go here)
        return cfg;
    }

} // namespace smallm