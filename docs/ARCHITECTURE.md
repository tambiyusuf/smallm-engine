# Architecture

`smallm-engine` is a from-scratch inference engine for GGUF language models,
written in modern C++. It is not a wrapper around llama.cpp: every layer — file
loading, tensor handling, operators, the compute backend, and the execution of a
model — is built here directly, to keep full control over how a model runs on the
hardware.

This document explains the project's structure, the responsibility of each
module, and how to extend the engine along its three natural axes: new
architectures, new backends, and new operators.

---

## Design philosophy

- **No black boxes.** Every layer is implemented and understood, from the byte
  layout of the GGUF file up to the token that comes out.
- **Layered and decoupled.** Pure math (operators) knows nothing about hardware.
  Hardware execution (backend) knows nothing about model architecture. Model
  architecture knows nothing about which backend runs underneath. Each layer
  depends only on the interface below it.
- **Config-driven, architecture-agnostic.** No dimension is ever hard-coded. Every
  size (layer count, head count, hidden size, rope base, …) is read from the
  model file, so the same code runs any model of a supported architecture family,
  at any scale.
- **Extensible without over-engineering.** Interfaces are minimal and grow only
  when a real need appears. A new architecture, backend, or operator is added by
  writing an isolated file, not by touching existing code.

---

## Directory layout

The source tree mirrors the header tree exactly: for every `include/smallm/X`
there is a matching `src/X`. This makes finding the implementation of any
declaration trivial.

```
include/smallm/            src/
├── core/                  ├── core/
│   ├── gguf.h             │   ├── gguf.cpp
│   ├── ops.h             │   ├── ops.cpp
│   └── tensor.h           │   └── tensor.cpp
├── backend/               ├── backend/
│   ├── common/            │   └── cpu_backend.cpp
│   │   └── backend.h      │
│   └── cpu_backend.h      │
├── config/                ├── config/
│   ├── common/            │   ├── common/
│   │   ├── config.h       │   │   └── config.cpp
│   │   └── meta_read.h    │   └── qwen2_config.cpp
│   └── qwen2_config.h     │
└── model/                 └── model/
    ├── common/                └── qwen2_model.cpp
    │   └── model.h
    └── qwen2_model.h
```

### Module responsibilities

| Module        | Responsibility                                                                 |
|---------------|--------------------------------------------------------------------------------|
| `core/`       | Foundational, hardware- and architecture-independent building blocks.          |
| `core/gguf`   | Parse a GGUF file via `mmap` (zero-copy): header, metadata, tensor table.       |
| `core/tensor` | Represent a tensor and dequantize its weights (F32, Q8_0, Q4_0) to float.       |
| `core/ops`    | Reference scalar math: `matmul`, `rmsnorm`, `softmax`, `silu`, `rope`.          |
| `backend/`    | Abstract compute backend: *where* the operators run (CPU now, GPU later).       |
| `config/`     | Read architecture parameters from metadata into a typed config struct.          |
| `model/`      | Assemble weights and run the forward pass for a specific architecture.          |

---

## Layered architecture

The engine is organized as a stack of layers. Each depends only on the one below.

```
            ┌───────────────────────────────────────────┐
            │  model/   (Qwen2Model : Model)             │  architecture
            │  embedding → layers → norm → logits        │  assembly
            └───────────────────────────────────────────┘
                              │ calls
            ┌───────────────────────────────────────────┐
            │  backend/  (Backend interface)             │  where compute runs
            │  CPUBackend  (CUDABackend, ROCmBackend …)  │
            └───────────────────────────────────────────┘
                              │ delegates to
            ┌───────────────────────────────────────────┐
            │  core/ops   (reference scalar math)        │  pure math
            └───────────────────────────────────────────┘
                              │ operates on
            ┌───────────────────────────────────────────┐
            │  core/tensor + core/gguf                   │  data
            │  mmap'd weights, dequantized to float      │
            └───────────────────────────────────────────┘
```

Two independent extension mechanisms use inheritance:

- **`Model` → `Qwen2Model`** — a pure-virtual `forward()` defines the contract;
  each architecture implements it. The runtime talks only to `Model`, never to a
  concrete type.
- **`Backend` → `CPUBackend`** — a pure-virtual set of operations defines the
  contract; each backend implements it. The forward pass talks only to `Backend`,
  never to raw ops, so swapping in a GPU backend does not touch model code.

Config uses the same base/derived pattern (`ModelConfig` → `Qwen2Config`), held by
`unique_ptr` to avoid object slicing.

---

## Forward pass, end to end

For one token at position `pos`, `Qwen2Model::forward` produces logits over the
full vocabulary:

1. **Embedding lookup** — copy the token's row from the embedding table.
2. **For each of the N transformer layers:**
   - Pre-norm (RMSNorm), then **attention**: project Q/K/V (with Qwen2's biases),
     apply RoPE to Q and K, write K/V into the per-layer KV cache, attend causally
     over all cached positions using **grouped-query attention (GQA)**, then the
     output projection. Add the result back (residual).
   - Pre-norm (RMSNorm), then **feed-forward (SwiGLU)**: `silu(gate(x)) * up(x)`,
     then the down projection. Add the result back (residual).
3. **Final norm** (RMSNorm).
4. **Output projection** to logits (falls back to the tied embedding table if the
   model has no separate output weight).

The KV cache is pre-allocated to the model's context length and indexed by
position, so generation advances one token at a time without reallocation. This
structure is the foundation later optimizations (e.g. prefix caching) build on.

---

## How to extend

### Add a new architecture (e.g. Llama)

Most transformers share the same skeleton; a new architecture is usually the same
building blocks arranged slightly differently.

1. `config/llama_config.h` + `src/config/llama_config.cpp` — define
   `LlamaConfig : ModelConfig` (add fields only if Llama needs any beyond the
   core) and `read_llama_config`, filling it from `"llama."`-prefixed metadata.
2. Add one branch to `read_config` in `config/common/config.cpp`:
   `else if (arch == "llama") return read_llama_config(model);`
3. `model/llama_model.h` + `src/model/llama_model.cpp` — define
   `LlamaModel : Model`, reusing the shared operators. Only the parts that differ
   from Qwen2 (e.g. no attention bias) change.

Nothing else is touched. Existing architectures keep working.

### Add a new backend (e.g. CUDA)

1. `backend/cuda_backend.h` + `src/backend/cuda_backend.cu` — define
   `CUDABackend : Backend`, implementing each operation with device kernels and
   whatever memory management the device needs.
2. Select it when building the model (a backend choice passed to the model, rather
   than a hard-coded `CPUBackend`).

The forward pass does not change: it already calls through the `Backend`
interface, so it runs on whatever backend it is given.

### Add a new operator (e.g. GELU)

Operators are written **just in time** — each one is added and tested together
with the first architecture that uses it, so no untested code accumulates.

1. Declare it in `core/ops.h` and implement it in `core/ops.cpp` (the reference
   scalar version), with a small invariant-based test.
2. Add it to the `Backend` interface and implement it in each backend
   (`CPUBackend` can delegate to the new `ops` function).

---

## Current status

Implemented and verified:

- GGUF loader — mmap, zero-copy, full header/metadata/tensor-table parsing.
- Dequantization — F32, Q8_0, Q4_0.
- Operators — matmul, rmsnorm, softmax, silu, rope, each individually tested.
- Backend abstraction — `Backend` interface + `CPUBackend`.
- Config — architecture-dispatched reader (`ModelConfig` base + `Qwen2Config`).
- Qwen2 forward pass — end-to-end logits on CPU, producing well-formed output on
  Qwen2.5-0.5B.

---

## Türkçe Özet

`smallm-engine`, GGUF formatındaki dil modellerini sıfırdan çalıştıran, modern
C++ ile yazılmış bir çıkarım (inference) motorudur. llama.cpp'nin bir sarmalayıcısı
değildir: dosya okuma, tensor yönetimi, operatörler, hesaplama backend'i ve modelin
çalıştırılması dahil her katman doğrudan burada inşa edilmiştir. Amaç, modelin
donanımda nasıl koştuğu üzerinde tam kontrol sağlamaktır.

**Tasarım felsefesi:** Siyah kutu yok; her katman anlaşılarak yazıldı. Katmanlar
birbirinden ayrık: saf matematik (operatörler) donanımı bilmez, backend mimariyi
bilmez, mimari de altındaki backend'i bilmez. Hiçbir boyut koda gömülü değildir —
katman sayısı, head sayısı, gizli boyut gibi her değer model dosyasından okunur,
böylece aynı kod desteklenen bir mimarinin her boyutunu çalıştırır.

**Dizin yapısı:** `src/` ağacı `include/` ağacını birebir yansıtır. Modüller:
`core/` (gguf, ops, tensor — temel taşlar), `backend/` (işlemlerin nerede koştuğu —
şimdi CPU, ileride GPU), `config/` (mimari parametrelerini metadata'dan okuma),
`model/` (belirli bir mimarinin forward pass'i).

**Katmanlı mimari:** İki bağımsız genişleme mekanizması kalıtım kullanır:
`Model → Qwen2Model` (her mimari kendi `forward`'ını yazar) ve
`Backend → CPUBackend` (her backend kendi işlemlerini sağlar). Config de aynı
taban/türev desenini (`ModelConfig → Qwen2Config`) kullanır ve slicing'i önlemek
için `unique_ptr` ile tutulur.

**Nasıl genişletilir:**
- *Yeni mimari (ör. Llama):* kendi config ve model dosyalarını ekle, `read_config`'e
  bir dal ekle. Mevcut hiçbir şeye dokunulmaz.
- *Yeni backend (ör. CUDA):* `Backend`'i uygulayan yeni bir sınıf yaz. Forward pass
  değişmez, çünkü zaten arayüz üzerinden çağırıyor.
- *Yeni operatör (ör. GELU):* önce `core/ops`'a referans (test edilmiş) sürümünü,
  sonra `Backend` arayüzüne ekle. Operatörler, onları ilk kullanan mimariyle
  birlikte yazılır — kullanılmayan kod birikmez.

**Mevcut durum:** GGUF okuma, dequantize (F32/Q8_0/Q4_0), operatörler, backend
soyutlaması, config okuma ve Qwen2 forward pass tamamlandı ve doğrulandı;
Qwen2.5-0.5B üzerinde uçtan uca çalışıp geçerli logit üretiyor.
