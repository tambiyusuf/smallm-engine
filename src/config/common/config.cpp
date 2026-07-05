//
// Created by tambiyusuf on 5.07.2026.
//
#include "smallm/config/common/config.h"
#include "smallm/config/qwen2_config.h"

#include <stdexcept>

#include "smallm/config/common/meta_read.h"

namespace smallm {

    std::unique_ptr<ModelConfig> read_config(const GGUFModel& model) {
        // reuse the shared helper instead of hand-rolling the lookup
        std::string arch = meta::get<std::string>(model, "general.architecture");

        if (arch == "qwen2") {
            return read_qwen2_config(model);
        }
        // future: else if (arch == "llama") return read_llama_config(model);

        throw std::runtime_error("config: unsupported architecture: " + arch);
    }

} // namespace smallm