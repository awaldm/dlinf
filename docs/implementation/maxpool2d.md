# MaxPool2d

The current MaxPool2d validation target is the ResNet-18 stem pool:

```text
input:   [64, 112, 112]
kernel:  3
stride:  2
padding: 1
output:  [64, 56 * 56]
```

## Direct

`maxpool2d_direct` scans each output window and writes the maximum value for
each channel and spatial position. The implementation uses the same row-major
`[channels, height * width]` layout as Conv2d and BatchNorm2d.

The operation has no learned parameters, so the benchmark record reports an
empty `weight_shape`.

## Validation

`make test-maxpool` compares `maxpool2d_direct` against the exported PyTorch
`maxpool` output.

The same implementation is used inside `resnet18_direct`.
