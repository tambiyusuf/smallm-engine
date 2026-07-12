//
// Created by tambiyusuf on 6.07.2026.
//
#pragma once

#include "smallm/model/common/model.h"
#include "smallm/tokenizer/bpe_tokenizer.h"
#include "smallm/runtime/sampler/common/sampler.h"
#include "smallm/runtime/generation_stats.h"

#include <cstdint>
#include <string>

namespace smallm {

    class Generator {
    public:
        Generator(Model& model, const Tokenizer& tokenizer, Sampler& sampler);

        std::string generate(const std::string& prompt, uint32_t max_new_tokens);

        // stats from the most recent generate() call
        const GenerationStats& last_stats() const { return stats_; }

    private:
        Model& model_;
        const Tokenizer& tokenizer_;
        Sampler& sampler_;
        std::vector<uint32_t> cached_tokens_;

        GenerationStats stats_;   // filled by each generate() call
    };

} // namespace smallm