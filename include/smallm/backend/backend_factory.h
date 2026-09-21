#pragma once

#include "smallm/backend/common/backend.h"

#include <memory>
#include <string>

namespace smallm {

    enum class BackendKind {
        Cpu,
        Cuda,
    };

    // parses "cpu" or "cuda" for CLI --backend
    BackendKind parse_backend_kind(const std::string& name);

    // builds the requested compute backend; cuda throws if SMALLM_CUDA was not built
    std::unique_ptr<Backend> create_backend(BackendKind kind);

} // namespace smallm
