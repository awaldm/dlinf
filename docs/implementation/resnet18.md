# ResNet-18

Full TorchVision ResNet-18 inference pipeline implemented as a single composed
function.

```text
conv1 -> bn1 -> relu -> maxpool
-> layer1 (2 identity blocks)
-> layer2 (projection + identity)
-> layer3 (projection + identity)
-> layer4 (projection + identity)
-> avgpool -> fc
```

## Current Implementation

| Function | Status | Notes |
|---|---|---|
| `resnet18_direct` | validated and benchmark-wired | Composes all existing operators. Uses direct Conv2D, direct BatchNorm2D, direct MaxPool2D, direct AvgPool2D, and Eigen linear. |

The implementation loads all weight tensors from a `WeightArchive` by name and
chains the existing layer/block operators. Internal helper functions
encapsulate identity and projection BasicBlock calls to keep the main pipeline
readable.

## Validation Target

```text
artifacts/resnet18/resnet18_full_golden.elw
```

It contains:

| Tensor | Shape | Meaning |
|---|---|---|
| `resnet18.input` | `[3, 224, 224]` | deterministic input image |
| `resnet18.expected` | `[1000]` | PyTorch output (class logits) |

Run:

```bash
make test-resnet18
```

The current tolerance is `2e-5`.

## Benchmark Target

`make bench-models` runs the `resnet18 / resnet18_direct` case. It validates
against the full-model golden output before timing, like the smaller operator
and block cases.

The same case can be run directly:

```bash
./build/bench_kernels --case resnet18 --warmup 1 --iterations 3
```

The checked-in local laptop artifact may lag behind the harness. Treat the JSONL
file under `benchmarks/results/` as a saved measurement, not as the complete set
of cases that the current benchmark binary can execute.
