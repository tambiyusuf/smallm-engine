//
// Created by tambiyusuf on 4.07.2026.
//
#include "smallm/cli/cli.h"
#include "smallm/i18n/messages.h"

#include <iostream>

int main(int argc, char** argv) {
    try {
        return smallm::cli::run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << smallm::i18n::message(smallm::i18n::MessageId::ErrGeneric) << ": "
                  << e.what() << "\n";
        return 1;
    }
}
