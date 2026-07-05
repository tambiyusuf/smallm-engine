//
// Created by tambiyusuf on 6.07.2026.
//
#include "smallm/tokenizer/bpe_tokenizer.h"

#include <algorithm>
#include <stdexcept>

namespace smallm {

// GPT-2 byte-level mapping: every byte becomes a printable unicode code point.
// this reproduces the standard bytes_to_unicode() table used by GPT-2 BPE.
void BPETokenizer::build_byte_maps() {
    std::vector<uint8_t> bs;
    // printable ascii ranges that map to themselves
    for (int b = '!'; b <= '~'; ++b) bs.push_back(static_cast<uint8_t>(b));
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(static_cast<uint8_t>(b));
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(static_cast<uint8_t>(b));

    std::vector<int> cs(bs.begin(), bs.end());

    // remaining bytes get mapped to code points starting at 256, in order
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), static_cast<uint8_t>(b)) == bs.end()) {
            bs.push_back(static_cast<uint8_t>(b));
            cs.push_back(256 + n);
            ++n;
        }
    }

    // encode each code point as utf-8 and build both directions
    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t byte = bs[i];
        int cp = cs[i];

        // encode code point cp to a utf-8 string
        std::string u;
        if (cp < 0x80) {
            u.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            u.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            u.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            u.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            u.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            u.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }

        byte_to_unicode_[byte] = u;
        unicode_to_byte_[u] = byte;
    }
}

BPETokenizer::BPETokenizer(const GGUFModel& model) {
    build_byte_maps();

    // load vocab tokens
    auto tok = model.metadata.find("tokenizer.ggml.tokens");
    if (tok == model.metadata.end()) {
        throw std::runtime_error("tokenizer: missing tokenizer.ggml.tokens");
    }
    const auto* tokens = std::get_if<std::vector<std::string>>(&tok->second.data);
    if (!tokens) {
        throw std::runtime_error("tokenizer: tokens metadata is not a string array");
    }

    id_to_token_ = *tokens;
    for (uint32_t i = 0; i < id_to_token_.size(); ++i) {
        token_to_id_[id_to_token_[i]] = i;
    }

    // load merge rules, ranked by their order in the list
    auto mrg = model.metadata.find("tokenizer.ggml.merges");
    if (mrg != model.metadata.end()) {
        if (const auto* merges = std::get_if<std::vector<std::string>>(&mrg->second.data)) {
            for (int32_t rank = 0; rank < static_cast<int32_t>(merges->size()); ++rank) {
                // each merge entry is "tokenA tokenB"; key it as-is for lookup
                merge_rank_[(*merges)[rank]] = rank;
            }
        }
    }

    // special tokens
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

// decode: map each id to its token string, convert the byte-level unicode back
// to raw bytes, and concatenate
std::string BPETokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::string out;
    for (uint32_t id : ids) {
        if (id >= id_to_token_.size()) continue;   // skip out-of-range ids
        const std::string& tok = id_to_token_[id];

        // walk the token's utf-8 chars, mapping each back to its original byte
        size_t i = 0;
        while (i < tok.size()) {
            // determine utf-8 char length
            unsigned char c = tok[i];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;

            std::string ch = tok.substr(i, len);
            auto it = unicode_to_byte_.find(ch);
            if (it != unicode_to_byte_.end()) {
                out.push_back(static_cast<char>(it->second));
            } else {
                out += ch;  // fallback: emit as-is
            }
            i += len;
        }
    }
    return out;
}

    // encode: byte-level BPE. map bytes to their unicode form, then greedily merge
    // adjacent pairs by rank until no ranked pair remains, and map pieces to ids.
    std::vector<uint32_t> BPETokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> out;
    if (text.empty()) return out;

    // step 1: turn the raw bytes into their GPT-2 byte-level unicode pieces.
    // each piece starts as one byte-mapped token.
    std::vector<std::string> pieces;
    for (unsigned char c : text) {
        auto it = byte_to_unicode_.find(c);
        if (it != byte_to_unicode_.end()) pieces.push_back(it->second);
    }
    if (pieces.empty()) return out;

    // step 2: repeatedly merge the highest-priority adjacent pair.
    while (pieces.size() > 1) {
        int best_rank = -1;
        size_t best_i = 0;

        // scan all adjacent pairs, find the one with the lowest (best) rank
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            std::string pair = pieces[i] + " " + pieces[i + 1];
            auto it = merge_rank_.find(pair);
            if (it != merge_rank_.end()) {
                if (best_rank == -1 || it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }
        }

        // no mergeable pair left -> done
        if (best_rank == -1) break;

        // merge the winning pair into a single piece
        pieces[best_i] = pieces[best_i] + pieces[best_i + 1];
        pieces.erase(pieces.begin() + best_i + 1);
    }

    // step 3: map each final piece to its vocab id
    for (const std::string& p : pieces) {
        auto it = token_to_id_.find(p);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        }
        // pieces not in vocab are dropped; a full impl would byte-fallback here
    }
    return out;
}
} // namespace smallm