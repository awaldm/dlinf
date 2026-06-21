# dlinf

C++/Eigen inference framework with PyTorch weight export tooling.

The current implementation loads public TorchVision ResNet-18 weights, exports
PyTorch reference tensors, and validates selected C++ inference operators
against those references.

Implemented operator variants include:

- `linear_naive` and `linear_eigen`,
- `conv2d_naive_direct` and `conv2d_im2col_eigen`,
- `batchnorm2d_direct`,
- Eigen-backed in-place ReLU.

## Layout

```text
tools/audit_resnet18.py        ResNet-18 state_dict auditor and exporter
docs/weight_archive_format.md  dlinf flat weight archive specification
include/dlinf/                 C++17 public headers
src/                           C++ demo and loader implementation
benchmarks/bench_kernels.cpp   JSONL kernel benchmark runner
```

Documentation is available through MkDocs:

```bash
uv sync --group dev
uv run mkdocs serve
```

The public docs cover validation, implementation details, performance output,
and the `.elw` weight archive format.

## Audit ResNet-18

Create the Python tooling environment with uv:

```bash
uv sync
```

Then run the auditor:

```bash
uv run python tools/audit_resnet18.py --output-dir artifacts/resnet18
```

This writes:

- `artifacts/resnet18/resnet18_audit.json`
- `artifacts/resnet18/resnet18_imagenet1k_v1.elw`

Use `--weights none` if you only want shapes from an untrained architecture and
do not want TorchVision to download public ImageNet weights.

Export the PyTorch golden fixture for the final ResNet-18 `fc` layer:

```bash
uv run python tools/export_linear_golden.py --output-dir artifacts/resnet18
```

This writes:

- `artifacts/resnet18/fc_golden.elw`
- `artifacts/resnet18/fc_golden.json`

Export the PyTorch golden fixture for the first ResNet-18 `conv1` layer:

```bash
uv run python tools/export_conv_golden.py --output-dir artifacts/resnet18
```

This writes:

- `artifacts/resnet18/conv1_golden.elw`
- `artifacts/resnet18/conv1_golden.json`

Export the PyTorch golden fixture for `conv1 -> bn1`:

```bash
uv run python tools/export_conv_bn_golden.py --output-dir artifacts/resnet18
```

This writes:

- `artifacts/resnet18/conv1_bn1_golden.elw`
- `artifacts/resnet18/conv1_bn1_golden.json`

Export the PyTorch golden fixture for the first identity BasicBlock,
`layer1.0`:

```bash
uv run python tools/export_basicblock_golden.py --output-dir artifacts/resnet18
```

This writes:

- `artifacts/resnet18/layer1_0_basicblock_golden.elw`
- `artifacts/resnet18/layer1_0_basicblock_golden.json`

## Build And Run

Eigen is header-only, but the build expects an installed Eigen3 package.

On Arch/Omarchy:

```bash
sudo pacman -S eigen3
```

The supported C++ build path is the root `Makefile`:

```bash
make
./build/dlinf_demo artifacts/resnet18/resnet18_imagenet1k_v1.elw
```

Run the validation tests:

```bash
make test-linear
make test-conv2d
make test-conv-bn
```

Run the current kernel benchmark harness:

```bash
make bench-kernels
```

If Eigen is installed in a non-standard location, pass the header directory:

```bash
make EIGEN_INCLUDE=/path/to/eigen3
```
