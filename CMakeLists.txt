cmake_minimum_required(VERSION 3.16)
project(smallm_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SMALLM_CUDA "Build with NVIDIA CUDA backend" OFF)

# keep the build clean and strict while the codebase is small
add_compile_options(-Wall -Wextra -O2)
add_compile_options(-Wall -Wextra -mavx2 -mfma)
find_package(OpenMP REQUIRED)

set(SMALLM_SOURCES
        src/main.cpp

        # core
        src/core/gguf.cpp
        src/core/tensor.cpp
        src/core/ops.cpp
        src/core/quant_q4_k.cpp

        # config
        src/config/common/config.cpp
        src/config/qwen2_config.cpp
        src/config/llama_config.cpp

        # model
        src/model/qwen2_model.cpp
        src/model/llama_model.cpp
        src/model/model_factory.cpp

        # i18n
        src/i18n/locale.cpp
        src/i18n/messages.cpp

        # cli
        src/cli/cli.cpp

        # tokenizer
        src/tokenizer/bpe_tokenizer.cpp
        src/tokenizer/tokenizer_factory.cpp
        src/tokenizer/sp_tokenizer.cpp

        # runtime
        src/runtime/sampler/greedy_sampler.cpp
        src/runtime/sampler/temperature_sampler.cpp
        src/runtime/sampler/top_k_sampler.cpp
        src/runtime/sampler/top_p_sampler.cpp
        src/runtime/generator.cpp

        # backend
        src/backend/cpu_backend.cpp
        src/backend/backend_factory.cpp
)

if(SMALLM_CUDA)
    enable_language(CUDA)
    list(APPEND SMALLM_SOURCES src/backend/cuda_backend.cu)
endif()

add_executable(smallm ${SMALLM_SOURCES})

target_link_libraries(smallm PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(smallm PRIVATE include)

if(SMALLM_CUDA)
    target_compile_definitions(smallm PRIVATE SMALLM_CUDA=1)
    set_target_properties(smallm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(smallm PRIVATE CUDA::cudart)
endif()
