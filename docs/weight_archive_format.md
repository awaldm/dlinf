# Eigen-Learn Weight Archive Format

The Stage 1 export format is a single little-endian binary file with extension
`.elw` (`Eigen-Learn Weights`). It is intentionally flat and dependency-free so
the C++ loader can parse it with the standard library and expose tensor payloads
as `Eigen::Map` views.

## Design Goals

- **Zero-copy tensor access**: tensor data is stored as contiguous raw bytes and
  aligned to 64-byte boundaries for cache-line friendly reads.
- **Self-describing**: every tensor has a fixed-size record containing its name,
  dtype, rank, shape, byte offset, and byte size.
- **Simple C++ parser**: no JSON, Protobuf, or reflection library is required in
  the inference runtime.
- **PyTorch-compatible layout**: tensors are exported in contiguous C-order.
  ResNet convolution weights keep PyTorch's `[out_channels, in_channels, kH, kW]`
  layout.

## File Layout

```text
+----------------------+  offset 0
| FileHeader (64 B)    |
+----------------------+  offset records_offset
| TensorRecord[N]      |  each record is 224 B
+----------------------+  offset data_offset, 64-byte aligned
| Tensor payload bytes |
| ...                  |
+----------------------+
```

All integer fields are little-endian.

## FileHeader

| Field | Type | Meaning |
|---|---:|---|
| `magic` | `char[8]` | ASCII `ELWT0001` |
| `version` | `uint32` | Format version, currently `1` |
| `endian_tag` | `uint32` | `0x01020304`; rejects byte-swapped files |
| `header_bytes` | `uint32` | Always `64` |
| `record_bytes` | `uint32` | Always `224` |
| `tensor_count` | `uint64` | Number of tensor records |
| `records_offset` | `uint64` | Absolute byte offset of first record |
| `data_offset` | `uint64` | Absolute byte offset of first payload byte |
| `file_bytes` | `uint64` | Expected archive size |
| `reserved` | `uint64` | Must be zero |

## TensorRecord

| Field | Type | Meaning |
|---|---:|---|
| `name` | `char[128]` | UTF-8 state_dict key, NUL padded |
| `dtype` | `uint32` | See dtype table below |
| `rank` | `uint32` | Number of valid entries in `shape` |
| `offset` | `uint64` | Absolute byte offset of payload |
| `nbytes` | `uint64` | Payload size in bytes |
| `elem_size` | `uint32` | Bytes per element |
| `flags` | `uint32` | Bit 0: C-contiguous |
| `shape` | `uint64[8]` | Tensor dimensions, padded with zeros |

Current dtype codes:

| Code | Dtype |
|---:|---|
| `1` | `float32` |
| `2` | `float64` |
| `3` | `int64` |
| `4` | `int32` |
| `5` | `uint8` |
| `6` | `float16` |

Stage 1 C++ supports `float32` tensors for Eigen math. Other dtypes are retained
in the archive so the auditor remains faithful to PyTorch state_dict contents;
for ResNet-18 this mainly preserves BatchNorm `num_batches_tracked` counters.

## Why Not Protobuf?

Protobuf is useful when the schema will evolve across teams or languages, but it
adds a runtime dependency and still leaves large tensor blobs as byte arrays.
For this project, the senior signal is the runtime memory model, not schema
tooling. A fixed binary table keeps the hot path inspectable, aligned, and easy
to memory-map later.

The auditor also writes JSON for human inspection. Treat JSON as a report, not
as the runtime ingestion format.

## Format Ideas Considered

The exporter currently writes two different artifacts because they serve
different jobs:

- `resnet18_audit.json` is for humans. It answers "what is inside this public
  model?" and "which layers do I need to implement?"
- `resnet18_imagenet1k_v1.elw` is for the C++ engine. It answers "where are the
  bytes for `fc.weight`, and how should I interpret them?"

The split is deliberate. JSON is excellent for inspection, but it is not a good
runtime tensor format: parsing is slower, numbers are textual, and the C++
runtime would need to allocate and convert values before doing math. The `.elw`
archive stores raw tensor bytes directly, so the loader can point Eigen at the
payload.

Other reasonable formats:

| Format | What it is good for | Why not the Stage 1 default |
|---|---|---|
| ONNX | Full model graph interchange: layers, weights, operators, metadata | Too high-level for this project; it hides much of the loader and operator work |
| TorchScript / `.pt` | Staying inside PyTorch's runtime ecosystem | Pulls the project back toward PyTorch instead of a standalone C++ engine |
| safetensors | Safe, established tensor-only weight storage | Good alternative, but still adds format conventions and a parser dependency |
| Protobuf / FlatBuffers | Stable cross-language schemas | Useful for teams, but tensor payloads are still large byte blobs |
| NPZ / HDF5 | Convenient scientific storage | More runtime dependency and less direct control over layout |
| Custom flat binary | Alignment, offsets, raw bytes, simple C++ parser | Requires us to own versioning and validation |

For Eigen-Learn, the custom flat archive is a conscious portfolio choice. It is
not meant to beat ONNX as a general ecosystem standard. It is meant to show the
systems concerns that matter in an inference runtime:

- binary compatibility,
- byte offsets,
- dtype handling,
- tensor shape metadata,
- contiguous storage,
- alignment,
- eventually `mmap`-style loading.

The long-term practical compromise could be:

1. Use ONNX or safetensors as an optional import path.
2. Convert that public format into `.elw`.
3. Keep `.elw` as the internal runtime format optimized for the Eigen engine.

## How The Archive Is Used In C++

The C++ loader does not "understand ResNet" yet. It only understands a table of
named tensors. That is the right first step.

For example, if the archive contains:

```text
fc.weight  shape=[1000, 512] dtype=float32 offset=...
fc.bias    shape=[1000]      dtype=float32 offset=...
```

then C++ can ask:

```cpp
auto weight = archive.tensor_f32("fc.weight");
auto bias = archive.tensor_f32("fc.bias");
```

The returned object is a view into the archive's byte buffer. It does not copy
the tensor into a second owner. Once the data pointer and shape are available,
Eigen can wrap the pointer with `Eigen::Map` and perform matrix/vector math.

This is why the archive stores PyTorch tensors in contiguous C-order. It makes
the transition from:

```text
PyTorch state_dict tensor
```

to:

```text
raw archive bytes -> C++ tensor view -> Eigen::Map -> math kernel
```

as direct as possible.

## Next Implementation Steps

The next three steps are about proving the pipeline one piece at a time. We do
not try to implement all of ResNet-18 immediately.

### 3. Implement The First Real Layer: `Linear`

Start with the final ResNet-18 classifier because it is just dense matrix math.
ResNet-18 ends with:

```text
fc.weight: [1000, 512]
fc.bias:   [1000]
```

For one input feature vector:

```text
input:  [512]
output: [1000]
```

the math is:

```text
output = weight * input + bias
```

Expanded:

```text
output[0] = sum(weight[0, j] * input[j]) + bias[0]
output[1] = sum(weight[1, j] * input[j]) + bias[1]
...
output[999] = sum(weight[999, j] * input[j]) + bias[999]
```

This is the smallest useful C++ inference layer because it exercises:

- loading named tensors from `.elw`,
- interpreting a 2D tensor as an Eigen matrix,
- interpreting a 1D tensor as an Eigen vector,
- doing real model math with public weights.

The C++ implementation should look conceptually like:

```cpp
Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    weight(fc_weight.data(), 1000, 512);

Eigen::Map<const Eigen::VectorXf> bias(fc_bias.data(), 1000);
Eigen::Map<const Eigen::VectorXf> input(input_data, 512);

Eigen::VectorXf output = weight * input + bias;
```

This does not yet run a real image through ResNet. It proves the runtime can
load model weights and execute one real layer correctly.

### 4. Add A PyTorch Golden Test

A golden test compares the C++ result against a known-good PyTorch result.

The Python side should:

1. Load the same ResNet-18 weights.
2. Create a deterministic input vector of shape `[512]`.
3. Run only the PyTorch `fc` layer.
4. Export:
   - the input vector,
   - the expected output vector.

The C++ side should:

1. Load `fc.weight` and `fc.bias` from `.elw`.
2. Load the exported input vector.
3. Run the C++ `Linear` implementation.
4. Compare each output value with the PyTorch expected output.

The check is not exact bit-for-bit equality. Floating-point math can differ
slightly depending on compiler flags, Eigen vectorization, BLAS choices, and
operation order. The useful assertion is:

```text
abs(cpp_output[i] - torch_output[i]) < tolerance
```

with a tolerance around `1e-5` for float32.

This step is important because it creates a validation pattern for every future
operator. Without golden tests, it is easy to write a convolution that "looks
right" but has a layout, padding, stride, or channel-order bug.

### 5. Move To Conv2d And BatchNorm Folding

After `Linear`, the next serious ResNet operation is convolution.

PyTorch stores Conv2d weights as:

```text
[out_channels, in_channels, kernel_height, kernel_width]
```

For `conv1.weight` in ResNet-18:

```text
[64, 3, 7, 7]
```

That means:

- 64 output channels,
- 3 input channels,
- 7x7 kernel per input channel,
- one output pixel is a weighted sum over `3 * 7 * 7` input values.

The direct convolution implementation has to handle:

- channel layout,
- padding,
- stride,
- nested loops over output pixels,
- nested loops over input channels and kernel positions,
- cache behavior.

BatchNorm appears after most convolutions in ResNet-18. During inference,
BatchNorm is not really a dynamic layer anymore. Its running statistics are
fixed:

```text
running_mean
running_var
gamma / weight
beta / bias
eps
```

For inference, this:

```text
conv -> batchnorm
```

can be folded into a single adjusted convolution:

```text
conv_with_adjusted_weight_and_bias
```

The formula per output channel is:

```text
scale = gamma / sqrt(running_var + eps)
folded_weight = conv_weight * scale
folded_bias = (conv_bias - running_mean) * scale + beta
```

If the convolution originally has no bias, treat `conv_bias` as zero.

This matters because ResNet-18 has many Conv+BatchNorm pairs. Folding reduces
the runtime operator inventory and makes the C++ engine simpler:

```text
Conv2d -> BatchNorm -> ReLU
```

becomes:

```text
FoldedConv2d -> ReLU
```

That is both faster and easier to validate. The Python auditor/exporter can
eventually write folded weights directly, or it can write original weights and
let C++ fold them at load time. For Stage 1 and early Stage 2, folding in Python
is simpler because PyTorch can be used as the reference implementation.

## Recommended Near-Term Order

1. Run the ResNet-18 auditor and inspect `resnet18_audit.json`.
2. Build the C++ demo and confirm it can read the `.elw` archive.
3. Implement `Linear` using `fc.weight` and `fc.bias`.
4. Add a PyTorch golden test for `Linear`.
5. Add a direct, simple Conv2d implementation.
6. Add Conv2d golden tests for `conv1`.
7. Add BatchNorm folding and compare folded output against PyTorch
   `conv -> batchnorm`.

