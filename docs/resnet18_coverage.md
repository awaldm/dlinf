# ResNet-18 Coverage

This page tracks which parts of TorchVision ResNet-18 have exported golden
fixtures, C++ validation, and benchmark coverage.

The diagram is intentionally staged instead of module-by-module. It shows
validation/export coverage at a readable scale, not a full neural-network graph
executor.

```mermaid
flowchart TB
    input["input image"]
    convbn["conv1 + bn1"]
    relupool["relu + maxpool<br/>maxpool exporter added"]
    layer1["layer1<br/>layer1.0 validated &amp; benchmarked<br/>layer1.1 planned"]
    layer2["layer2<br/>projection block exported<br/>identity block planned"]
    layer3["layer3<br/>projection block planned<br/>identity block planned"]
    layer4["layer4<br/>projection block planned<br/>identity block planned"]
    head["head<br/>avgpool exported<br/>fc validated"]


    input --> convbn --> relupool --> layer1 --> layer2 --> layer3 --> layer4 --> head

    classDef validated fill:#166534,stroke:#22c55e,color:#ffffff;
    classDef exported fill:#92400e,stroke:#f59e0b,color:#ffffff;
    classDef planned fill:#374151,stroke:#9ca3af,color:#ffffff;

    class convbn,layer1,head validated;
    class relupool,layer2 exported;
    class input,layer3,layer4 planned;
```

Legend:

| Color | Meaning |
|---|---|
| Green | At least one part of the stage is validated in C++. |
| Yellow | Golden exporter exists, but C++ validation is still planned. |
| Gray | Planned only. |

## Nomenclature

TorchVision ResNet-18 has a bespoke front end, a repeated block body, and a
small classification head:

```text
input
-> conv1 -> bn1 -> relu -> maxpool
-> layer1 -> layer2 -> layer3 -> layer4
-> avgpool -> flatten -> fc
```

The front end is often called the **stem**. It is not a BasicBlock. In
TorchVision ResNet-18, the stem starts with a `7x7` stride-2 convolution, then
BatchNorm, ReLU, and MaxPool.

The recurring body consists of four **layer groups**:

| Layer group | Blocks | Role |
|---|---|---|
| `layer1` | `layer1.0`, `layer1.1` | First BasicBlock group; keeps 64 channels. |
| `layer2` | `layer2.0`, `layer2.1` | Downsamples and increases to 128 channels. |
| `layer3` | `layer3.0`, `layer3.1` | Downsamples and increases to 256 channels. |
| `layer4` | `layer4.0`, `layer4.1` | Downsamples and increases to 512 channels. |

ResNet-18 uses two kinds of BasicBlock:

| Block kind | Examples | Skip path |
|---|---|---|
| Identity BasicBlock | `layer1.0`, `layer1.1`, `layer2.1`, `layer3.1`, `layer4.1` | Adds the original input tensor directly. |
| Projection BasicBlock | `layer2.0`, `layer3.0`, `layer4.0` | Uses a `1x1` downsample convolution plus BatchNorm on the skip path. |

The current BasicBlock exporter targets `layer1.0` because it is an identity
BasicBlock: it exercises the recurring block structure without needing the
projection/downsample skip path yet.

## Coverage Table

| ResNet-18 part | Golden exporter | C++ validation | Benchmark coverage | Status |
|---|---|---|---|---|
| `fc` | `tools/export_linear_golden.py` | `make test-linear` | `make bench-kernels` | Validated and benchmarked |
| `conv1` | `tools/export_conv_golden.py` | `make test-conv2d` | `make bench-kernels` | Validated and benchmarked |
| `conv1 -> bn1` | `tools/export_conv_bn_golden.py` | `make test-conv-bn` | `make bench-kernels` | Validated and benchmarked |
| `layer1.0` identity BasicBlock | `tools/export_basicblock_golden.py` | `make test-basicblock` | `make bench-kernels` | Validated and benchmarked |
| `maxpool` (stem) | `tools/export_maxpool_golden.py` | Planned | Planned | Golden exporter added |
| `avgpool` (head) | `tools/export_avgpool_golden.py` | Planned | Planned | Golden exporter added |
| Projection BasicBlocks | `tools/export_projection_basicblock_golden.py` | Planned | Planned | Golden exporter added |
| `layer1.1` identity BasicBlock | reuse `export_basicblock_golden.py --block layer1.1` | Planned | Planned | Reuses existing exporter |
| Full ResNet-18 | Planned | Planned | Planned | After BasicBlock coverage |

## Export Pattern

Golden fixture archives contain only the deterministic block input and expected
PyTorch output. Model parameters stay in the full ResNet-18 weight archive:

```text
artifacts/resnet18/resnet18_imagenet1k_v1.elw
```

This keeps each fixture small and makes the C++ tests prove that they can load
weights from the same archive used by other validation targets.
