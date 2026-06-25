# dlinf

`dlinf` (for deep learning inference, duh) is a small C++/Eigen inference validation and benchmark harness.

It executes TorchVision ResNet-18 in a small C++ runtime, validates the C++
outputs against PyTorch-generated golden fixtures, and records benchmark results
for selected operator and block implementations.

The goal is not to provide a production inference library. I aim to make
the inference path inspectable: weights, tensor layouts, operator composition,
numerical agreement, and timing are all explicit.


## What This Is

- A controlled C++ inference-runtime study.
- A validation harness for comparing C++ outputs against PyTorch.
- A small benchmark suite for inference operators, blocks, and full ResNet-18.
- A way to inspect implementation choices such as direct convolution vs. im2col + Eigen.

## What This Is Not

- A production inference runtime.
- A TensorRT, ONNX Runtime, or OpenVINO replacement.
- A new neural-network architecture.
- A high performance library (it is mostly unvectorized C++).


## Core Principle

Before benchmarking a C++ inference path, verify that it matches the PyTorch
reference numerically.

For each measured operator or block, `dlinf` exports a PyTorch golden fixture,
runs the equivalent C++ implementation on the same inputs and weights, checks
the maximum absolute error, and only then records timing results.


## Quick Start/Reproduction
This repo is mainly an implementation and benchmark study. The commands below reproduce the validation and benchmark artifacts.

```bash
uv sync
uv run python tools/audit_resnet18.py --output-dir artifacts/resnet18
uv run python tools/export_all_golden.py --output-dir artifacts/resnet18
```
This writes:

- `artifacts/resnet18/resnet18_audit.json`
- `artifacts/resnet18/resnet18_imagenet1k_v1.elw`

Use `--weights none` if you only want shapes from an untrained architecture and
do not want TorchVision to download public ImageNet weights.



```bash
make
make test
make bench-kernels-save
make plot-benchmarks
```

Documentation is available through MkDocs:

```bash
uv sync --group dev
uv run mkdocs serve
```

The public docs cover validation, implementation details, performance output,
and the `.elw` weight archive format.


## Example Benchmark Output

Benchmark results are written as JSONL:

```json
{"op":"resnet18","impl":"resnet18_eigen","precision":"fp32","threads":1,
"median_ms":123.4,"max_abs_error":0.00002}
```
See docs/performance.md for the current example run and generated charts.


## Prerequisites

- for python: uv
- for c++: Eigen


