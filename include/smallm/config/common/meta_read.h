//
// Created by tambiyusuf on 5.07.2026.
//
#pragma once

#include "smallm/core/gguf.h"
#include "smallm/i18n/messages.h"

#include <string>

namespace smallm::meta {

    // pulls a value of the requested type from metadata, or throws if missing/mismatched
    template <typename T>
    T get(const GGUFModel& model, const std::string& key) {
        auto it = model.metadata.find(key);
        if (it == model.metadata.end()) {
            i18n::throw_error(i18n::MessageId::ErrConfigMissingKey, key);
        }
        const T* val = std::get_if<T>(&it->second.data);
        if (!val) {
            i18n::throw_error(i18n::MessageId::ErrConfigWrongType, key);
        }
        return *val;
    }

    // same, but returns a fallback instead of throwing when the key is absent
    template <typename T>
    T get_or(const GGUFModel& model, const std::string& key, T fallback) {
        auto it = model.metadata.find(key);
        if (it == model.metadata.end()) return fallback;
        const T* val = std::get_if<T>(&it->second.data);
        return val ? *val : fallback;
    }

} // namespace smallm::meta