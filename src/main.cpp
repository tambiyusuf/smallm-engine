//
// Created by tambiyusuf on 4.07.2026.
//
#include "smallm/core/gguf.h"
#include "smallm/model/qwen2_model.h"
#include <iostream>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: smallm <model.gguf>\n"; return 1; }
    try {
        smallm::GGUFModel model = smallm::load_gguf(argv[1]);
        smallm::Qwen2Model qwen(std::move(model));
        std::cout << "model ready.\n";

        // feed a single token at position 0, get logits, find the argmax
        uint32_t token = 9707;  // arbitrary test token id
        std::vector<float> logits = qwen.forward(token, 0);

        uint32_t best = 0;
        for (uint32_t i = 1; i < logits.size(); ++i)
            if (logits[i] > logits[best]) best = i;

        std::cout << "logits size: " << logits.size() << "\n";
        std::cout << "argmax token id: " << best
                  << "  (logit " << logits[best] << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}