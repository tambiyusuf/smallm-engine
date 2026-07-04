//
// Created by tambiyusuf on 4.07.2026.
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>

namespace smallm {

    // value types a metadata entry can hold, matching the GGUF spec's type ids
    enum class GGUFType : uint32_t {
        UINT8   = 0,
        INT8    = 1,
        UINT16  = 2,
        INT16   = 3,
        UINT32  = 4,
        INT32   = 5,
        FLOAT32 = 6,
        BOOL    = 7,
        STRING  = 8,
        ARRAY   = 9,
        UINT64  = 10,
        INT64   = 11,
        FLOAT64 = 12,
    };

    // a single metadata value; arrays are homogeneous, so each array kind is its own vector type
    struct GGUFValue {
        std::variant <
            uint8_t, int8_t, uint16_t, int16_t,
            uint32_t, int32_t, uint64_t, int64_t,
            float, double, bool, std::string,
            std::vector<int32_t>,
            std::vector<uint32_t>,
            std::vector<float>,
            std::vector<std::string>
        > data;
    };


    // describes one tensor: where it lives in the file and its shape
    struct GGUFTensorInfo {
        std::string name;
        std::vector<uint64_t> dims;   // shape, fastest-moving dimension first
        uint32_t type;                // quantization / dtype id from the spec
        uint64_t offset;              // byte offset into the tensor data blob
    };

    // the whole parsed file: header counts, metadata, tensor table
    struct GGUFModel {
        uint32_t version = 0;
        uint64_t tensor_count = 0;
        uint64_t metadata_count = 0;

        std::unordered_map<std::string, GGUFValue> metadata;
        std::vector<GGUFTensorInfo> tensors;

        // start of the raw tensor data, past all the headers
        uint64_t tensor_data_offset = 0;
    };

    // opens a GGUF file via mmap and fills a GGUFModel; throws on malformed input
    GGUFModel load_gguf(const std::string& path);

} // namespace smallm