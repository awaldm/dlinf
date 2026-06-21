# BatchNorm2d

`batchnorm2d_direct` implements inference-time BatchNorm for tensors represented
as `[channels, elements]`.

For each channel and element:

```text
output[c, i] = gamma[c] * (input[c, i] - mean[c]) / sqrt(var[c] + eps) + beta[c]
```

The default `batchnorm2d` wrapper currently calls `batchnorm2d_direct`.

## Validation

`make test-conv-bn` validates BatchNorm through the ResNet-18 `conv1 -> bn1`
fixture. The test checks both Conv2d paths followed by `batchnorm2d_direct`:

```text
conv2d_naive_direct -> batchnorm2d_direct
conv2d_im2col_eigen -> batchnorm2d_direct
```
