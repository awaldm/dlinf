# BasicBlock

The current block implementation targets the identity-skip ResNet-18
BasicBlock used by `layer1.0`.

```text
conv1 -> batchnorm -> relu -> conv2 -> batchnorm -> residual_add -> relu
```

## Current Implementation

| Function | Status | Notes |
|---|---|---|
| `basicblock_direct` | validated | Uses direct Conv2D, direct BatchNorm2D, in-place ReLU, and checked elementwise residual add. |

The implementation covers identity residual blocks only. The other blocks are projection/downsample blocks and need a separate skip-path
implementation with `1x1` convolution and BatchNorm.

## Validation Target

The current golden fixture is:

```text
artifacts/resnet18/layer1_0_basicblock_golden.elw
```

It contains:

| Tensor | Shape | Meaning |
|---|---:|---|
| `layer1.0.input` | `[64, 56, 56]` | deterministic block input |
| `layer1.0.expected` | `[64, 56, 56]` | PyTorch output for `model.get_submodule("layer1.0")` |

The C++ test loads weights from the full ResNet-18 archive and compares the
composed block output against the PyTorch fixture:

```bash
make test-basicblock
```

The current tolerance is `2e-5`.
