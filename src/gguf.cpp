//
// Created by tambiyusuf on 4.07.2026.
//
#include "smallm/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace smallm {

namespace {

// a tiny cursor over the mmap'd bytes; every read advances the position
struct Reader {
    const uint8_t* base;   // start of the mapping
    size_t size;           // total mapped bytes
    size_t pos = 0;        // current offset

    // copies `n` raw bytes into `out`, then moves forward; guards against overrun
    void read_raw(void* out, size_t n) {
        if (pos + n > size) throw std::runtime_error("gguf: read past end of file");
        std::memcpy(out, base + pos, n);
        pos += n;
    }

    // reads a single fixed-size value (uint32, uint64, float, ...)
    template <typename T>
    T read() {
        T v;
        read_raw(&v, sizeof(T));
        return v;
    }

    // GGUF strings are a uint64 length followed by that many raw bytes
    std::string read_string() {
        uint64_t len = read<uint64_t>();
        if (pos + len > size) throw std::runtime_error("gguf: string past end of file");
        std::string s(reinterpret_cast<const char*>(base + pos), len);
        pos += len;
        return s;
    }
};

// reads one metadata value of the given type into a GGUFValue
GGUFValue read_value(Reader& r, GGUFType type);

// reads an array: an element type, a count, then that many elements
GGUFValue read_array(Reader& r) {
    GGUFType elem_type = static_cast<GGUFType>(r.read<uint32_t>());
    uint64_t count = r.read<uint64_t>();

    GGUFValue out;

    switch (elem_type) {
        case GGUFType::INT32: {
            std::vector<int32_t> v(count);
            for (uint64_t i = 0; i < count; ++i) v[i] = r.read<int32_t>();
            out.data = std::move(v);
            break;
        }
        case GGUFType::UINT32: {
            std::vector<uint32_t> v(count);
            for (uint64_t i = 0; i < count; ++i) v[i] = r.read<uint32_t>();
            out.data = std::move(v);
            break;
        }
        case GGUFType::FLOAT32: {
            std::vector<float> v(count);
            for (uint64_t i = 0; i < count; ++i) v[i] = r.read<float>();
            out.data = std::move(v);
            break;
        }
        case GGUFType::STRING: {
            std::vector<std::string> v(count);
            for (uint64_t i = 0; i < count; ++i) v[i] = r.read_string();
            out.data = std::move(v);
            break;
        }
        default:
            throw std::runtime_error("gguf: unsupported array element type");
    }
    return out;
}

// dispatches a single value read based on its declared type
GGUFValue read_value(Reader& r, GGUFType type) {
    GGUFValue out;
    switch (type) {
        case GGUFType::UINT8:   out.data = r.read<uint8_t>();  break;
        case GGUFType::INT8:    out.data = r.read<int8_t>();   break;
        case GGUFType::UINT16:  out.data = r.read<uint16_t>(); break;
        case GGUFType::INT16:   out.data = r.read<int16_t>();  break;
        case GGUFType::UINT32:  out.data = r.read<uint32_t>(); break;
        case GGUFType::INT32:   out.data = r.read<int32_t>();  break;
        case GGUFType::UINT64:  out.data = r.read<uint64_t>(); break;
        case GGUFType::INT64:   out.data = r.read<int64_t>();  break;
        case GGUFType::FLOAT32: out.data = r.read<float>();    break;
        case GGUFType::FLOAT64: out.data = r.read<double>();   break;
        case GGUFType::BOOL:    out.data = static_cast<bool>(r.read<uint8_t>()); break;
        case GGUFType::STRING:  out.data = r.read_string();    break;
        case GGUFType::ARRAY:   out = read_array(r);           break;
        default:
            throw std::runtime_error("gguf: unknown metadata value type");
    }
    return out;
}

} // namespace

GGUFModel load_gguf(const std::string& path) {
    // open the file read-only and grab its size
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("gguf: cannot open file: " + path);

    struct stat st{};
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw std::runtime_error("gguf: fstat failed");
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    // map the whole file; the OS pages it in on demand, nothing is copied up front
    void* mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // the mapping stays valid after closing the fd
    if (mapping == MAP_FAILED) throw std::runtime_error("gguf: mmap failed");

    Reader r{static_cast<const uint8_t*>(mapping), file_size, 0};

    // header: 4-byte magic must spell "GGUF"
    char magic[4];
    r.read_raw(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        munmap(mapping, file_size);
        throw std::runtime_error("gguf: bad magic, not a GGUF file");
    }

    GGUFModel model;
    model.version        = r.read<uint32_t>();
    model.tensor_count   = r.read<uint64_t>();
    model.metadata_count = r.read<uint64_t>();

    // metadata: each entry is key string, value type, then the value
    for (uint64_t i = 0; i < model.metadata_count; ++i) {
        std::string key = r.read_string();
        GGUFType type = static_cast<GGUFType>(r.read<uint32_t>());
        model.metadata[key] = read_value(r, type);
    }

    // tensor table: name, dim count, dims, type id, offset into the data blob
    model.tensors.reserve(model.tensor_count);
    for (uint64_t i = 0; i < model.tensor_count; ++i) {
        GGUFTensorInfo t;
        t.name = r.read_string();

        uint32_t n_dims = r.read<uint32_t>();
        t.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) t.dims[d] = r.read<uint64_t>();

        t.type   = r.read<uint32_t>();
        t.offset = r.read<uint64_t>();
        model.tensors.push_back(std::move(t));
    }

    // tensor data starts after the headers, aligned up to the file's alignment
    uint32_t alignment = 32;  // GGUF default unless metadata overrides it
    auto it = model.metadata.find("general.alignment");
    if (it != model.metadata.end()) {
        if (auto* a = std::get_if<uint32_t>(&it->second.data)) alignment = *a;
    }
    uint64_t unaligned = r.pos;
    model.tensor_data_offset = (unaligned + alignment - 1) & ~(uint64_t(alignment) - 1);

    // note: we deliberately leak the mapping for now; tensor access will need it later
    return model;
}

} // namespace smallm