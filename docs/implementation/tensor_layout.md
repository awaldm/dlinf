# Tensor Layout

The current C++ Conv2d and BatchNorm2d interfaces use row-major Eigen matrices
for comparison views:

```text
[channels, height * width]
```

For ResNet-18 `conv1`:

```text
PyTorch output shape: [64, 112, 112]
C++ comparison view:  [64, 12544]
```

This keeps the current operator interfaces small while still matching PyTorch
reference outputs exactly for the exported fixtures.

Weights retain their PyTorch layout in the `.elw` archive:

```text
Linear weight: [out_features, in_features]
Conv2d weight: [out_channels, in_channels, kernel_height, kernel_width]
```

The implementation maps these raw row-major buffers into Eigen views where
needed.
