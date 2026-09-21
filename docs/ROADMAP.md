# Roadmap

smallm-engine is a from-scratch GGUF inference stack in C++. It is not a
llama.cpp wrapper. The goal is a readable codebase with a first-class CPU path,
NVIDIA CUDA support, and AMD ROCm support over time.

## Done (baseline)

- GGUF mmap loader, Qwen2 and Llama forward on CPU
- Quantized matmul (Q4_0, Q8_0, Q6_K, Q4_K / Q4_K_M on CPU)
- CLI: `generate`, `inspect`, `bench`
- Backend interface and CPU implementation
- Optional CUDA backend (Q4_0 matmul on GPU, weight upload cache)

## Phase 1 — CPU stability

- GitHub Actions CPU build
- Backend factory and `--backend cpu|cuda`
- Broader quant coverage as needed (Q5_K, IQ types)
- Chat templates and streaming output

## Phase 2 — CUDA (NVIDIA)

- Expand GPU kernels (Q8_0, Q6_K, Q4_K)
- GPU rmsnorm, rope, silu, softmax
- Keep weights on device; reduce host sync per token
- Document driver and CUDA toolkit versions in README

## Phase 3 — ROCm (AMD)

- HIP backend implementing the same `Backend` interface
- Port CUDA kernels where practical
- Multi-GPU and institutional deployment guides

## Phase 4 — Community and product

- HTTP server (OpenAI-compatible subset)
- CONTRIBUTING guide and labeled issues
- Benchmark table per release (CPU vs CUDA vs ROCm)

## Out of scope (for now)

- Training and fine-tuning
- Wrapping or forking llama.cpp
- Cloud-only managed inference
