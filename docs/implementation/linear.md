# Linear

The `Linear` layer is validated against the final ResNet-18 classifier:

```text
input:  [512]
weight: [1000, 512]
bias:   [1000]
output: [1000]
```

## Direct

`linear_naive` is the scalar-loop baseline:

```text
for out
  sum = bias[out]
  for in
    sum += weight[out, in] * input[in]
```

It is useful as a readable baseline and as a check that the weight layout is
being interpreted correctly.

## Eigen

`linear_eigen` maps PyTorch's row-major weight tensor as an Eigen matrix and
uses Eigen's matrix-vector multiply:

```text
output = weight * input + bias
```

The default `linear` wrapper currently calls `linear_eigen`.

## Validation

`make test-linear` compares `linear_naive`, `linear_eigen`, and the default
wrapper against the exported PyTorch `fc` output.
