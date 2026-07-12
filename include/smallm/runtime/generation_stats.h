//
// Created by tambiyusuf on 9.07.2026.
//
#pragma once

#include <cstdint>

namespace smallm {

    // timing and token counts for one generate() call. extended over time with
    // finer metrics (per-layer timing, memory, cache-hit ratio, etc.) as needed.
    struct GenerationStats {
        uint32_t prompt_tokens    = 0;   // total tokens in the prompt
        uint32_t reused_tokens    = 0;   // skipped via prefix caching
        uint32_t prefill_tokens   = 0;   // actually prefilled (prompt minus reused)
        uint32_t generated_tokens = 0;   // tokens produced in the decode loop

        double prefill_ms = 0.0;
        double decode_ms  = 0.0;

        // throughput helpers
        double prefill_tokens_per_sec() const {
            return prefill_ms > 0.0 ? prefill_tokens / (prefill_ms / 1000.0) : 0.0;
        }
        double decode_tokens_per_sec() const {
            return decode_ms > 0.0 ? generated_tokens / (decode_ms / 1000.0) : 0.0;
        }
    };

} // namespace smallm