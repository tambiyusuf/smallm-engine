# Contributing

Thanks for helping improve smallm-engine.

## Build

```bash
cmake -B build -DCMAKE_CXX_COMPILER=g++
cmake --build build -j
```

With CUDA (optional):

```bash
cmake -B build -DSMALLM_CUDA=ON
cmake --build build -j
```

Place GGUF models under `models/` (not committed). See README for CLI examples.

## Code style

- C++20, match existing naming and module layout (`include/smallm` mirrors `src/`)
- Keep comments short and in English
- Prefer small, focused PRs with a clear test story (CLI command or bench output)

## Areas where help is welcome

| Label / area | Examples |
|--------------|----------|
| `area:quant` | New GGUF types, faster CPU kernels |
| `area:cuda` | NVIDIA kernels, memory residency |
| `area:rocm` | HIP port, AMD tuning |
| `area:runtime` | Generator, samplers, chat templates |
| `area:docs` | Architecture notes, benchmarks |

## ROCm

We plan a HIP backend after CUDA stabilizes. If you work on AMD GPUs, open an
issue or PR against `docs/ROADMAP.md` Phase 3 with your hardware and ROCm version.

## Pull requests

1. Describe what changed and how you tested it
2. Keep unrelated refactors out of the same PR
3. Ensure the CPU CI build passes

## License

By contributing, you agree that your contributions are licensed under the same
license as the project (see LICENSE).
