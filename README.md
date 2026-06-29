# dlinf

`dlinf` is a small C++/Eigen inference validation and benchmark artifact for
TorchVision ResNet-18.

It loads exported PyTorch weights and golden tensors, runs the same computation
in a small C++ runtime, checks numerical agreement, and records benchmark
results for selected operator, block, and full-model paths.

The goal is not to provide a production inference library. The goal is to make
a neural-network inference path inspectable end to end: tensor layout, weight
loading, operator composition, correctness checks, and timing output are all
explicit.


## What This Does

- exporting TorchVision ResNet-18 weights and reference outputs
- loading model weights into a small C++ runtime
- implementing inference operators directly and with Eigen-backed variants
- validating C++ outputs against PyTorch golden fixtures before benchmarking
- recording benchmark results as machine-readable JSONL
- documenting correctness, implementation choices, and performance claims

## What This Is Not

- A production inference runtime
- A TensorRT, ONNX Runtime, or OpenVINO replacement
- A new neural network architecture
- A high performance framework


## Core Principle

Before benchmarking a C++ inference path, verify that it matches the PyTorch reference numerically.

For each measured operator or block, `dlinf` exports a PyTorch golden fixture,
runs the equivalent C++ implementation on the same inputs and weights, checks
the maximum absolute error, and only then records timing results.

## Artifact Contract

Every meaningful benchmark in this repo should answer four questions:

| Question | Where it is answered |
|---|---|
| What computation is being run? | operator/block/model name and input shapes |
| What is the reference? | PyTorch-exported golden fixture |
| Does the C++ result match? | validation target and `max_abs_error` |
| What exactly was timed? | JSONL benchmark row with implementation, threads, flags, and host metadata |


## Reproduce The Artifact

This repo is mainly an implementation and benchmark study. The commands below
reproduce the reference artifacts, validation checks, benchmark output, and
generated plots.

Generate the PyTorch reference artifacts:

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



Build, validate, benchmark, and regenerate plots:

```bash
make
make test
make bench-kernels-save
make bench-models-save
make plot-benchmarks
```

Documentation is available through MkDocs:

```bash
uv sync --group dev
uv run mkdocs serve
```

The public docs cover validation, implementation details, performance output,
and the `.elw` weight archive format.


## Example Evidence Row

Benchmark results are written as JSONL. A row is meant to be read as an evidence
record: what was run, what implementation was timed, and what the error was
against the PyTorch reference.

```json
{"op":"resnet18","impl":"resnet18_direct","precision":"fp32","threads":1,
"median_ms":123.4,"max_abs_error":0.00002,"compiler_flags":"-O3 ...",
"cpu_model":"..."}
```

See docs/performance.md for the current example run and generated charts.


## Prerequisites

- for python: uv
- for c++: Eigen


