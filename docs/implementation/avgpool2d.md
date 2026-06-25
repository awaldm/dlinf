# AvgPool2d

The current AvgPool2d validation target is the ResNet-18 classification head:

```text
input:   [512, 7, 7]
kernel:  7
stride:  1
padding: 0
output:  [512, 1 * 1]
```

## Direct

`avgpool2d_direct` scans each output window and writes the mean value for each
channel and spatial position. For TorchVision ResNet-18 with a `[3, 224, 224]`
input, this is equivalent to the final `AdaptiveAvgPool2d((1, 1))` because the
feature map entering the head is `[512, 7, 7]`.

The operation has no learned parameters, so the benchmark record reports an
empty `weight_shape`.

## Validation

`make test-avgpool` compares `avgpool2d_direct` against the exported PyTorch
`avgpool` output.

The same implementation is used inside `resnet18_direct` before flattening and
the final `fc` layer.
