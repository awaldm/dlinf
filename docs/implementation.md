# Implementation

The framework currently focuses on a small set of ResNet-18 operators. Each
operator exposes explicit implementation names so tests and benchmarks can
state exactly what they measure.

## Modules

| Module | Implementations | Notes |
|---|---|---|
| [Linear](implementation/linear.md) | `linear_naive`, `linear_eigen` | final ResNet-18 classifier layer |
| [Conv2d](implementation/conv2d.md) | `conv2d_naive_direct`, `conv2d_im2col_eigen` | ResNet-18 `conv1` shape currently validated |
| [BatchNorm2d](implementation/batchnorm2d.md) | `batchnorm2d_direct` | inference-time channel normalization |
| [Tensor Layout](implementation/tensor_layout.md) | `[channels, height * width]` views | current C++ comparison layout |

Benchmarks call explicit implementation names rather than default wrappers.
This keeps the performance output readable as more implementations are added.
