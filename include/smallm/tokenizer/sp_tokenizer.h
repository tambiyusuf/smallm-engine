//
// Created by tambiyusuf on 13.07.2026.
//
#pragma once

#include "smallm/tokenizer/common/tokenizer.h"
#include "smallm/core/gguf.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace smallm {

    // SentencePiece-style BPE, as used by Llama. merges are chosen by token score
    // rather than by an explicit merge-rank list, and spaces are encoded as U+2581.
    class SPTokenizer : public Tokenizer {
    public:
        explicit SPTokenizer(const GGUFModel& model);

        std::vector<uint32_t> encode(const std::string& text) const override;
        std::string decode(const std::vector<uint32_t>& ids) const override;

        uint32_t vocab_size() const override {
            return static_cast<uint32_t>(id_to_token_.size());
        }
        int32_t bos_id() const override { return bos_id_; }
        int32_t eos_id() const override { return eos_id_; }

    private:
        std::vector<std::string> id_to_token_;
        std::vector<float> scores_;                          // one score per token
        std::unordered_map<std::string, uint32_t> token_to_id_;

        int32_t bos_id_ = -1;
        int32_t eos_id_ = -1;
    };

} // namespace smallm