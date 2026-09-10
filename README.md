<div align="center">

# Triton MuonAdamW

An easy-to-use, faster MuonAdamW optimizer implemented with custom Triton kernels.

</div>

---

## Overview

The objective of this project is to provide a faster MuonAdamW optimizer without requiring users to build a custom PyTorch wrapper every time. `MuonAdamW` exposes one optimizer API while internally routing parameters to the appropriate Triton optimizer.

`MuonAdamW` applies:

- Triton Muon to eligible 2D weight tensors.
- Triton AdamW to biases, normalization parameters, embeddings, output heads, and other non-Muon parameters.

The repository also includes a PyTorch-backed wrapper using `torch.optim.Muon` and fused `torch.optim.AdamW`. It is used as a matching reference implementation for benchmarking.

The Muon implementation uses three Triton matrix kernels for its Newton-Schulz transformation:

1. Compute the Gram matrix `X @ X.T`.
2. Apply the polynomial matrix transform.
3. Multiply the transformed matrix by `X` and write the result.

## Kernel Optimizations

The Triton kernels are optimized around the memory and launch behavior of the matrix operations:

- Grouped tile program IDs improve L2 cache reuse by keeping nearby matrix tiles close in the execution order.
- Symmetric matrix products compute one triangle and mirror the result, avoiding duplicate tile multiplications.
- The update work is fused inside the kernels, reducing intermediate tensors and extra global-memory passes.
- Matrix products use fixed-size tiles and masked edge loads/stores so partial tiles remain safe without separate cleanup kernels.
- AdamW updates load the parameter, gradient, and optimizer state together, then write the updated parameter and moments in the same kernel.

## Benchmark Results

Results below are reported as relative speed compared with the named baseline. A value above `1.0x` means the Triton implementation is faster than that baseline.

| Component | Triton implementation | Comparison baseline | Reported result |
| --- | --- | --- | ---: |
| AdamW | Triton AdamW | Naive / foreach AdamW | **2.65 - 2.70x** |
| AdamW | Triton AdamW | Fused PyTorch AdamW | **0.97 - 0.99x** |
| Muon | Triton Muon | Naive `torch.optim.Muon` | **1.33 - 1.35x** |
| Muon | Triton Muon | `torch.compile` + PyTorch Muon | **1.27 - 1.30x** |
| Hybrid | Triton MuonAdamW | PyTorch MuonAdamW wrapper | **1.24 - 1.26x** |

The hybrid comparison uses the same parameter split and optimizer configurations on both sides. AdamW uses the fused PyTorch implementation in the reference wrapper.

The custom optimizers were numerically compared against their PyTorch counterparts within these tolerances: AdamW used `atol=1e-4` and `rtol=1e-3`; Muon used `atol=1e-2` and `rtol=1e-2`. Muon uses bfloat16 intermediates, so its comparison uses looser tolerances.

## Project Tree

```text
Triton MuonAdamW/
├── MuonAdamW.py                 # Hybrid optimizer and PyTorch reference wrapper
├── adamw/
│   ├── triton_adam.py           # Triton AdamW kernel and optimizer
│   └── adam_benchmark.py        # AdamW benchmark
├── muon/
│   ├── triton_muon.py           # Triton Muon kernels and optimizer
│   └── muon_benchmark.py        # Muon benchmark
├── muon_adamw_benchmark.py      # Hybrid benchmark
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # Locked dependency versions
└── README.md
```

## Installation

The project targets Python 3.13 or newer and requires a CUDA-capable PyTorch and Triton environment.

```bash
uv sync
```

The benchmark scripts require a CUDA device and can be run with:

```bash
python adamw/adam_benchmark.py
python muon/muon_benchmark.py
python muon_adamw_benchmark.py
```

## Usage

The hybrid optimizer accepts `AdamConfig` and `MuonConfig` instances so the two optimizer branches can be configured independently. Only the values you want to change need to be provided; all other fields keep their defaults.

```python
from triton_muon_adamw import AdamConfig, MuonAdamW, MuonConfig

# `model` is an existing model on CUDA.
adam_config = AdamConfig(
	lr=3e-4,
)

muon_config = MuonConfig(
	lr=1e-3,
	ns_steps=6,
)

optimizer = MuonAdamW(
	model,
	adam_config=adam_config,
	muon_config=muon_config,
	parameter_split = "auto"
)
```

For transformer-style models that expose `get_input_embeddings()` and optionally `get_output_embeddings()`, `parameter_split="auto"` can be used instead of building explicit groups.

## Performance Note

All reported results hold only for models with more than 10 million parameters. Below that size, kernel launch overhead becomes a larger fraction of each optimization step, and the custom Triton kernels can underperform their PyTorch counterparts.
