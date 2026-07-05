//
// Created by tambiyusuf on 6.07.2026.
//
#pragma once

#include "smallm/core/gguf.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace smallm {

    // byte-level BPE tokenizer (GPT-2 style), as used by Qwen2.
    // loads its vocab and merge rules from GGUF metadata.
    class BPETokenizer {
    public:
        // builds the tokenizer from a loaded model's metadata
        explicit BPETokenizer(const GGUFModel& model);

        // text -> token ids
        std::vector<uint32_t> encode(const std::string& text) const;

        // token ids -> text
        std::string decode(const std::vector<uint32_t>& ids) const;

        uint32_t vocab_size() const { return static_cast<uint32_t>(id_to_token_.size()); }

        // special token ids read from metadata (bos/eos), -1 if absent
        int32_t bos_id() const { return bos_id_; }
        int32_t eos_id() const { return eos_id_; }

    private:
        // vocab: id <-> token string (tokens are in GPT-2 byte-level encoded form)
        std::vector<std::string> id_to_token_;
        std::unordered_map<std::string, uint32_t> token_to_id_;

        // merge rules: pair of token strings -> rank (lower rank = higher priority)
        std::unordered_map<std::string, int32_t> merge_rank_;

        int32_t bos_id_ = -1;
        int32_t eos_id_ = -1;

        // maps a raw byte to its GPT-2 byte-level unicode representation and back
        std::unordered_map<uint8_t, std::string> byte_to_unicode_;
        std::unordered_map<std::string, uint8_t> unicode_to_byte_;

        void build_byte_maps();
    };

} // namespace smallm