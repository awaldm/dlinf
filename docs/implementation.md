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
| [MaxPool2d](implementation/maxpool2d.md) | `maxpool2d_direct` | spatial downsampling, no learned parameters |
| [AvgPool2d](implementation/avgpool2d.md) | `avgpool2d_direct` | spatial averaging, no learned parameters |
| [BasicBlock](implementation/basicblock.md) | `basicblock_direct` | ResNet-18 `layer1.0` identity block validated |
| [Tensor Layout](implementation/tensor_layout.md) | `[channels, height * width]` views | current C++ comparison layout |

Benchmarks call explicit implementation names rather than default wrappers.
This keeps the performance output readable as more implementations are added.
