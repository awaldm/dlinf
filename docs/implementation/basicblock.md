# BasicBlock

The current block implementation covers TorchVision ResNet-18 BasicBlocks with
either an identity skip path or a projection/downsample skip path.

```text
conv1 -> batchnorm -> relu -> conv2 -> batchnorm -> residual_add -> relu
```

## Current Implementation

| Function | Status | Notes |
|---|---|---|
| `basicblock_direct` | validated and benchmarked | Uses direct Conv2D, direct BatchNorm2D, in-place ReLU, optional projection skip path, and checked elementwise residual add. |

For identity blocks, the residual path is the original input tensor. For
projection blocks, the residual path is `downsample.0` (`1x1` convolution)
followed by `downsample.1` (BatchNorm2d). The current tests cover `layer1.0`
as an identity block and `layer2.0` as a projection block.

## Validation Target

Current golden fixtures:

```text
artifacts/resnet18/layer1_0_basicblock_golden.elw
artifacts/resnet18/layer2_0_projection_basicblock_golden.elw
```

It contains:

| Tensor | Shape | Meaning |
|---|---:|---|
| `layer1.0.input` | `[64, 56, 56]` | identity block input |
| `layer1.0.expected` | `[64, 56, 56]` | PyTorch output for `model.get_submodule("layer1.0")` |
| `layer2.0.input` | `[64, 56, 56]` | projection block input |
| `layer2.0.expected` | `[128, 28, 28]` | PyTorch output for `model.get_submodule("layer2.0")` |

The C++ test loads weights from the full ResNet-18 archive and compares the
composed block output against the PyTorch fixture:

```bash
make test-basicblock
make test-projection-basicblock
```

The current tolerance is `2e-5`.
