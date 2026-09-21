#pragma once

#include <string>

namespace smallm::i18n {

    enum class Language {
        English,
        Turkish,
    };

    // active UI language for errors and CLI output
    Language current_language();
    void set_language(Language lang);

    // parse --lang and optional --save-lang from argv; strips those flags
    void apply_cli_language_flags(int& argc, char** argv, bool& save_language,
                                  bool& language_from_cli);

    // env SMALLM_LANG, then config file, then interactive setup on first run
    void init_locale();

    Language parse_language_tag(const std::string& tag);

    void save_language_preference(Language lang);
    bool load_language_preference(Language& out);

    void print_help();

} // namespace smallm::i18n
