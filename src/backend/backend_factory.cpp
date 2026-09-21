#include "smallm/backend/backend_factory.h"

#include "smallm/backend/cpu_backend.h"
#include "smallm/i18n/messages.h"

#ifdef SMALLM_CUDA
#include "smallm/backend/cuda_backend.h"
#endif

namespace smallm {

BackendKind parse_backend_kind(const std::string& name) {
    if (name == "cpu") {
        return BackendKind::Cpu;
    }
    if (name == "cuda") {
        return BackendKind::Cuda;
    }
    i18n::throw_error(i18n::MessageId::ErrUnknownBackend, name);
}

std::unique_ptr<Backend> create_backend(BackendKind kind) {
    switch (kind) {
    case BackendKind::Cpu:
        return std::make_unique<CPUBackend>();
    case BackendKind::Cuda:
#ifdef SMALLM_CUDA
        return std::make_unique<CUDABackend>();
#else
        i18n::throw_error(i18n::MessageId::ErrCudaNotBuilt);
#endif
    }
    i18n::throw_error(i18n::MessageId::ErrUnknownBackend, "?");
}

} // namespace smallm
