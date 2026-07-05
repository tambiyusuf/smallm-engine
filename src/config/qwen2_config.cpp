//
// Created by tambiyusuf on 5.07.2026.
//
#include "smallm/config/qwen2_config.h"
#include "smallm/config/common/meta_read.h"

namespace smallm {

    std::unique_ptr<Qwen2Config> read_qwen2_config(const GGUFModel& model) {
        auto cfg = std::make_unique<Qwen2Config>();

        cfg->architecture = "qwen2";
        const std::string p = "qwen2";  // metadata key prefix for this architecture

        // core structural fields
        cfg->n_layers       = meta::get<uint32_t>(model, p + ".block_count");
        cfg->embd_dim       = meta::get<uint32_t>(model, p + ".embedding_length");
        cfg->ffn_dim        = meta::get<uint32_t>(model, p + ".feed_forward_length");
        cfg->n_heads        = meta::get<uint32_t>(model, p + ".attention.head_count");
        cfg->context_length = meta::get<uint32_t>(model, p + ".context_length");

        // GQA: kv head count defaults to full head count on non-grouped models
        cfg->n_kv_heads     = meta::get_or<uint32_t>(model, p + ".attention.head_count_kv",
                                                     cfg->n_heads);

        // constants with sensible defaults
        cfg->rope_freq_base = meta::get_or<float>(model, p + ".rope.freq_base", 10000.0f);
        cfg->rms_eps        = meta::get_or<float>(model,
                                  p + ".attention.layer_norm_rms_epsilon", 1e-6f);

        // vocab size inferred from the token list length
        auto tok = model.metadata.find("tokenizer.ggml.tokens");
        if (tok != model.metadata.end()) {
            if (auto* v = std::get_if<std::vector<std::string>>(&tok->second.data)) {
                cfg->vocab_size = static_cast<uint32_t>(v->size());
            }
        }

        return cfg;
    }

} // namespace smallm