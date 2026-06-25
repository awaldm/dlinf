# Validation

The C++ operators are checked against PyTorch-generated reference tensors. The
Python tools export inputs, weights, and expected outputs into `.elw` archives.
The C++ tests load the same data and compare max absolute error against a fixed
tolerance.

## Reference Artifacts

| Artifact | Producer | Purpose |
|---|---|---|
| `resnet18_imagenet1k_v1.elw` | `tools/audit_resnet18.py` | TorchVision ResNet-18 weights |
| `fc_golden.elw` | `tools/export_linear_golden.py` | `fc` input and expected output |
| `conv1_golden.elw` | `tools/export_conv_golden.py` | `conv1` input and expected output |
| `conv1_bn1_golden.elw` | `tools/export_conv_bn_golden.py` | `conv1 -> bn1` input and expected output |
| `layer1_0_basicblock_golden.elw` | `tools/export_basicblock_golden.py` | `layer1.0` BasicBlock input and expected output |
| `maxpool_golden.elw` | `tools/export_maxpool_golden.py` | `maxpool` (stem) input and expected output |
| `avgpool_golden.elw` | `tools/export_avgpool_golden.py` | `avgpool` (head) input and expected output |
| `layer2_0_projection_basicblock_golden.elw` | `tools/export_projection_basicblock_golden.py` | `layer2.0` projection BasicBlock input and expected output |
| `resnet18_full_golden.elw` | `tools/export_resnet18_golden.py` | Full ResNet-18 `[3, 224, 224] -> [1000]` input and expected output |

## Test Commands

```bash
make test-linear
make test-conv2d
make test-conv-bn
make test-basicblock
make test-maxpool
make test-projection-basicblock
make test-avgpool
make test-resnet18
```

## Current Checks

| Test | Implementations checked | Reference |
|---|---|---|
| `test-linear` | `linear_naive`, `linear_eigen`, default `linear` | PyTorch `fc` |
| `test-conv2d` | `conv2d_naive_direct`, `conv2d_im2col_eigen`, default `conv2d` | PyTorch `conv1` |
| `test-conv-bn` | direct and im2col Conv2d paths followed by `batchnorm2d_direct` | PyTorch `conv1 -> bn1` |
| `test-basicblock` | `basicblock_direct` identity residual block | PyTorch `layer1.0` |
| `test-maxpool` | `maxpool2d_direct` | PyTorch `maxpool` |
| `test-avgpool` | `avgpool2d_direct` | PyTorch `avgpool` |
| `test-projection-basicblock` | `basicblock_direct` projection block | PyTorch `layer2.0` |
| `test-resnet18` | `resnet18_direct` full pipeline | PyTorch full ResNet-18 |

The broader coverage map is tracked in [ResNet-18 Coverage](resnet18_coverage.md).
