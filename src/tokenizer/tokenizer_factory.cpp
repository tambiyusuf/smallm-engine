//
// Created by tambiyusuf on 13.07.2026.
//
#include "smallm/tokenizer/common/tokenizer.h"
#include "smallm/tokenizer/bpe_tokenizer.h"
#include "smallm/tokenizer/sp_tokenizer.h"
#include "smallm/config/common/meta_read.h"

#include <stdexcept>

namespace smallm {

    std::unique_ptr<Tokenizer> build_tokenizer(const GGUFModel& model) {
        std::string kind = meta::get<std::string>(model, "tokenizer.ggml.model");

        if (kind == "gpt2")  return std::make_unique<BPETokenizer>(model);
        if (kind == "llama") return std::make_unique<SPTokenizer>(model);

        throw std::runtime_error("tokenizer: unsupported type: " + kind);
    }

} // namespace smallm