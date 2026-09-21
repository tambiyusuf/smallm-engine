#pragma once

#include <string>

namespace smallm::i18n {

    enum class MessageId {
        SetupChooseLanguage,

        ErrCannotOpenFile,
        ErrBadGgufMagic,
        ErrUnsupportedArchitecture,
        ErrUnsupportedTokenizer,
        ErrUnsupportedTensorType,
        ErrTensorNotFound,
        ErrConfigMissingKey,
        ErrConfigWrongType,
        ErrTokenizerMissingTokens,
        ErrTokenizerBadMetadata,
        ErrUnknownCliCommand,
        ErrMissingModelPath,
        ErrMissingSubcommand,
        ErrInvalidLanguage,
        ErrUnknownBackend,
        ErrCudaNotBuilt,
        ErrGeneric,

        HelpTitle,
        HelpUsage,
        HelpGenerate,
        HelpInspect,
        HelpBench,
        HelpLangFlag,

        LabelArchitecture,
        LabelVocab,
        LabelLayers,
        LabelContext,
        LabelTokenizer,
        LabelTensors,

        StatPrefill,
        StatDecode,
        StatCacheReused,
        StatTokensPerSec,
    };

    // returns the user-facing string for the active locale
    const char* message(MessageId id);

    // appends detail after the localized base message (e.g. file path)
    std::string format(MessageId id, const std::string& detail);

    [[noreturn]] void throw_error(MessageId id, const std::string& detail = "");

} // namespace smallm::i18n
