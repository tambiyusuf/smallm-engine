//
// Created by tambiyusuf on 13.07.2026.
//
#include "smallm/tokenizer/sp_tokenizer.h"

#include <stdexcept>
#include <limits>

namespace smallm {

namespace {
// SentencePiece marks word boundaries with U+2581 (lower one eighth block)
const std::string kSpaceMark = "\xE2\x96\x81";   // UTF-8 for U+2581
}

SPTokenizer::SPTokenizer(const GGUFModel& model) {
    auto tok = model.metadata.find("tokenizer.ggml.tokens");
    if (tok == model.metadata.end()) {
        throw std::runtime_error("tokenizer: missing tokenizer.ggml.tokens");
    }
    const auto* tokens = std::get_if<std::vector<std::string>>(&tok->second.data);
    if (!tokens) throw std::runtime_error("tokenizer: tokens is not a string array");

    id_to_token_ = *tokens;
    for (uint32_t i = 0; i < id_to_token_.size(); ++i) {
        token_to_id_[id_to_token_[i]] = i;
    }

    // scores drive merge decisions in SentencePiece
    auto sc = model.metadata.find("tokenizer.ggml.scores");
    if (sc != model.metadata.end()) {
        if (const auto* s = std::get_if<std::vector<float>>(&sc->second.data)) {
            scores_ = *s;
        }
    }
    if (scores_.size() != id_to_token_.size()) {
        // without scores we cannot rank merges; fall back to zeros
        scores_.assign(id_to_token_.size(), 0.0f);
    }

    auto read_special = [&](const std::string& key) -> int32_t {
        auto it = model.metadata.find(key);
        if (it == model.metadata.end()) return -1;
        if (const auto* v = std::get_if<uint32_t>(&it->second.data))
            return static_cast<int32_t>(*v);
        return -1;
    };
    bos_id_ = read_special("tokenizer.ggml.bos_token_id");
    eos_id_ = read_special("tokenizer.ggml.eos_token_id");
}

// encode: split into utf-8 characters (with spaces marked), then repeatedly merge
// the adjacent pair whose merged form has the highest vocab score.
std::vector<uint32_t> SPTokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> out;
    if (text.empty()) return out;

    // SentencePiece prefixes the text with a space mark
    std::string marked = kSpaceMark;
    for (char c : text) {
        if (c == ' ') marked += kSpaceMark;
        else          marked.push_back(c);
    }

    // split into utf-8 characters as the initial pieces
    std::vector<std::string> pieces;
    for (size_t i = 0; i < marked.size();) {
        unsigned char c = marked[i];
        size_t len = 1;
        if      ((c & 0xF8) == 0xF0) len = 4;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xE0) == 0xC0) len = 2;
        pieces.push_back(marked.substr(i, len));
        i += len;
    }

    // greedily merge the highest-scoring adjacent pair until none is left
    while (pieces.size() > 1) {
        float best_score = -std::numeric_limits<float>::infinity();
        size_t best_i = 0;
        bool found = false;

        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            std::string merged = pieces[i] + pieces[i + 1];
            auto it = token_to_id_.find(merged);
            if (it != token_to_id_.end()) {
                float s = scores_[it->second];
                if (s > best_score) {
                    best_score = s;
                    best_i = i;
                    found = true;
                }
            }
        }

        if (!found) break;

        pieces[best_i] = pieces[best_i] + pieces[best_i + 1];
        pieces.erase(pieces.begin() + best_i + 1);
    }

    // map final pieces to ids; unknown pieces fall back to per-byte tokens
    for (const std::string& p : pieces) {
        auto it = token_to_id_.find(p);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
            continue;
        }
        // byte fallback: SentencePiece encodes raw bytes as "<0xXX>"
        for (unsigned char b : p) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            auto bit = token_to_id_.find(buf);
            if (bit != token_to_id_.end()) out.push_back(bit->second);
        }
    }
    return out;
}

    // decode: concatenate token strings, turning space marks back into spaces and
    // byte tokens (<0xXX>) back into their raw bytes
    std::string SPTokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::string out;
    for (uint32_t id : ids) {
        if (id >= id_to_token_.size()) continue;
        const std::string& tok = id_to_token_[id];

        // byte tokens like "<0x0A>" decode to their raw byte
        if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x') {
            int byte = std::stoi(tok.substr(3, 2), nullptr, 16);
            out.push_back(static_cast<char>(byte));
            continue;
        }

        // every other token: copy its bytes, replacing each space mark with a space
        size_t i = 0;
        while (i < tok.size()) {
            if (tok.compare(i, kSpaceMark.size(), kSpaceMark) == 0) {
                out.push_back(' ');
                i += kSpaceMark.size();
            } else {
                out.push_back(tok[i]);
                ++i;
            }
        }
    }
    return out;
}
} // namespace smallm