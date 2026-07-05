//
// Created by tambiyusuf on 6.07.2026.
//
#pragma once

#include "smallm/model/common/model.h"
#include "smallm/tokenizer/bpe_tokenizer.h"
#include "smallm/runtime/sampler/common/sampler.h"

#include <cstdint>
#include <string>

namespace smallm {

    // drives token-by-token generation: encode prompt, run the model, sample,
    // decode, repeat until EOS or a token limit.
    class Generator {
    public:
        Generator(Model& model, const BPETokenizer& tokenizer, Sampler& sampler);

        std::string generate(const std::string& prompt, uint32_t max_new_tokens);

    private:
        Model& model_;
        const BPETokenizer& tokenizer_;
        Sampler& sampler_;
    };

} // namespace smallm
