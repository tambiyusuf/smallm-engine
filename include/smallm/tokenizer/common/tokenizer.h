//
// Created by tambiyusuf on 13.07.2026.
//
#pragma once

#include "smallm/core/gguf.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace smallm {

    // abstract tokenizer: text <-> token ids. concrete implementations cover the
    // different schemes models use (GPT-2 byte-level BPE, SentencePiece, ...).
    class Tokenizer {
    public:
        virtual ~Tokenizer() = default;

        virtual std::vector<uint32_t> encode(const std::string& text) const = 0;
        virtual std::string decode(const std::vector<uint32_t>& ids) const = 0;

        virtual uint32_t vocab_size() const = 0;
        virtual int32_t bos_id() const = 0;
        virtual int32_t eos_id() const = 0;
    };

    // picks the right tokenizer based on tokenizer.ggml.model in the metadata
    std::unique_ptr<Tokenizer> build_tokenizer(const GGUFModel& model);

} // namespace smallme smallm