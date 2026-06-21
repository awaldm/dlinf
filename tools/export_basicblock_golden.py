#!/usr/bin/env python3
"""Export a PyTorch golden fixture for one ResNet-18 identity BasicBlock.

This tool creates the next validation artifact for the C++ side of dlinf. It
does not export weights. Weights are expected to come from the full ResNet-18
weight archive produced by ``tools/audit_resnet18.py``. This exporter only
writes the deterministic input tensor for one BasicBlock and the corresponding
PyTorch output tensor.

Two files are produced:

``layer1_0_basicblock_golden.elw``
    Binary tensor archive consumed by C++ tests. It contains exactly two tensors
    by default: ``layer1.0.input`` and ``layer1.0.expected``.

``layer1_0_basicblock_golden.json``
    Human-readable contract and metadata for the fixture. It describes which
    TorchVision module path was executed, where the required weights live in the
    full ResNet-18 archive, which tensor names the C++ test should load, and
    which Torch/TorchVision versions produced the expected output.

The default block is ``layer1.0`` because it is the first ResNet-18 BasicBlock
with an identity residual path. Projection/downsample blocks such as
``layer2.0`` require extra skip-path logic and remain out of scope until the rest of the functionality is implemented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elw_archive import write_archive
from resnet18_common import load_resnet18


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a deterministic BasicBlock fixture export.

    ``--block`` uses PyTorch's dotted submodule path syntax, for example
    ``layer1.0``. The path is later resolved with ``model.get_submodule``.

    ``--height`` and ``--width`` describe the feature-map size entering the
    block, not the original input image size. For ResNet-18 ``layer1.0`` the
    natural ImageNet-size fixture shape is ``[64, 56, 56]``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Export deterministic PyTorch input/output for a ResNet-18 "
            "BasicBlock with an identity residual path."
        )
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet1k_v1", "default", "none"),
        default="imagenet1k_v1",
        help="TorchVision weight set to load.",
    )
    parser.add_argument(
        "--block",
        default="layer1.0",
        help="Dotted ResNet-18 BasicBlock path. Only identity blocks are supported for now.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/resnet18"),
        help="Directory for the golden .elw archive and JSON report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch RNG seed used for the deterministic block input tensor.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=56,
        help="Input feature-map height for the [C, H, W] fixture.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=56,
        help="Input feature-map width for the [C, H, W] fixture.",
    )
    return parser.parse_args()


def tuple_of_ints(values: Any) -> list[int]:
    """Convert PyTorch shape/stride tuples into plain JSON-serializable ints."""
    return [int(value) for value in values]


def block_file_stem(block_path: str) -> str:
    """Return the filename stem for a dotted ResNet module path.

    Tensor names keep the PyTorch dotted path, e.g. ``layer1.0.input``. File
    names replace dots with underscores so the fixture is easy to address from
    shells and Makefile targets, e.g. ``layer1_0_basicblock_golden.elw``.
    """
    return f"{block_path.replace('.', '_')}_basicblock_golden"


def require_identity_basicblock(block_path: str, block: Any) -> None:
    """Reject unsupported modules before writing a misleading fixture.

    The first C++ target is the plain ResNet BasicBlock:

    ``conv1 -> bn1 -> relu -> conv2 -> bn2 -> add(input) -> relu``

    This function checks that the selected PyTorch submodule has the expected
    BasicBlock attributes and no ``downsample`` projection. It also checks that
    the final output channel count can be added to the original input tensor.
    """
    required_attrs = ("conv1", "bn1", "relu", "conv2", "bn2")
    missing = [name for name in required_attrs if not hasattr(block, name)]
    if missing:
        raise SystemExit(
            f"{block_path} is not a supported ResNet BasicBlock; missing {missing}"
        )
    if getattr(block, "downsample", None) is not None:
        raise SystemExit(
            f"{block_path} has a downsample/projection residual path. "
            "Only identity BasicBlocks are supported by this exporter for now."
        )
    if int(block.conv1.in_channels) != int(block.conv2.out_channels):
        raise SystemExit(
            f"{block_path} does not preserve channel count across the identity path"
        )


def conv2d_report(block_path: str, name: str, conv: Any) -> dict[str, Any]:
    """Describe a Conv2d inside the JSON report.

    The returned dictionary is not written to the binary ``.elw`` file. It is
    documentation for the C++ test and an audit trail. The
    ``weight`` and ``bias`` fields name tensors in the full ResNet-18 weight
    archive, not tensors in this golden fixture archive.
    """
    return {
        "weight": f"{block_path}.{name}.weight",
        "bias": f"{block_path}.{name}.bias" if conv.bias is not None else None,
        "in_channels": int(conv.in_channels),
        "out_channels": int(conv.out_channels),
        "kernel_size": tuple_of_ints(conv.kernel_size),
        "stride": tuple_of_ints(conv.stride),
        "padding": tuple_of_ints(conv.padding),
        "dilation": tuple_of_ints(conv.dilation),
        "groups": int(conv.groups),
    }


def batchnorm2d_report(block_path: str, name: str, bn: Any) -> dict[str, Any]:
    """Describe a BatchNorm2d inside the JSON report.

    BatchNorm parameters and running statistics are loaded from the full
    ResNet-18 weight archive by C++ tests. The report names those tensors and
    records ``eps`` because it is part of the numerical contract.
    """
    return {
        "weight": f"{block_path}.{name}.weight",
        "bias": f"{block_path}.{name}.bias",
        "running_mean": f"{block_path}.{name}.running_mean",
        "running_var": f"{block_path}.{name}.running_var",
        "eps": float(bn.eps),
        "num_features": int(bn.num_features),
    }


def write_report(
    output_path: Path,
    torch_module: Any,
    torchvision_module: Any,
    weights_label: str,
    block_path: str,
    seed: int,
    archive_path: Path,
    manifest: list[dict[str, Any]],
    block: Any,
) -> None:
    """Write the JSON file that explains the binary fixture.

    ``manifest`` is returned by ``write_archive`` and describes the tensors that
    were actually written into the ``.elw`` file: tensor name, dtype, shape,
    byte offset, and payload size. This report wraps that low-level manifest
    with higher-level validation metadata:

    - ``fixture``: stable fixture identifier derived from the block path;
    - ``operation``: the TorchVision submodule lookup used to compute expected;
    - ``basicblock``: the ResNet block structure and parameter tensor names;
    - ``contract``: the exact input and expected tensor names for C++ tests.

    Keeping this as JSON makes the artifact inspectable without writing a
    binary archive reader, while the C++ test can still consume the binary
    ``.elw`` file.
    """
    stem = block_file_stem(block_path)
    report = {
        "fixture": f"resnet18_{stem}",
        "operation": f'torchvision.models.resnet18.get_submodule("{block_path}")',
        "weights": weights_label,
        "torch_version": torch_module.__version__,
        "torchvision_version": torchvision_module.__version__,
        "seed": seed,
        "archive": str(archive_path),
        "records": manifest,
        "basicblock": {
            "block": block_path,
            "residual_path": "identity",
            "conv1": conv2d_report(block_path, "conv1", block.conv1),
            "bn1": batchnorm2d_report(block_path, "bn1", block.bn1),
            "activation": "relu",
            "conv2": conv2d_report(block_path, "conv2", block.conv2),
            "bn2": batchnorm2d_report(block_path, "bn2", block.bn2),
        },
        "contract": {
            "input": f"{block_path}.input",
            "expected": f"{block_path}.expected",
            "meaning": (
                f"{block_path}.expected is computed by applying "
                f'model.get_submodule("{block_path}") to '
                f"{block_path}.input.unsqueeze(0), then squeezing the batch "
                "dimension."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    """Export the fixture and print a short shell-friendly summary."""
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise SystemExit("--height and --width must be positive")

    torch_module, torchvision_module, model, weights_label = load_resnet18(args.weights)
    try:
        block = model.get_submodule(args.block)
    except AttributeError:
        raise SystemExit(
            "This PyTorch version does not support Module.get_submodule."
        ) from None
    except Exception as exc:
        raise SystemExit(
            f"Could not find ResNet-18 block `{args.block}`: {exc}"
        ) from exc

    require_identity_basicblock(args.block, block)

    # The seed controls only the deterministic synthetic block input. It does
    # not affect pretrained weights when ``--weights`` selects a TorchVision
    # weight set.
    torch_module.manual_seed(args.seed)
    in_channels = int(block.conv1.in_channels)

    with torch_module.inference_mode():
        # C++ operators currently use channel-first tensors without a batch
        # dimension. PyTorch modules expect NCHW, so the batch dimension is
        # inserted for execution and removed again before writing the fixture.
        block_input = torch_module.randn(
            in_channels,
            args.height,
            args.width,
            dtype=torch_module.float32,
        )
        block_expected = block(block_input.unsqueeze(0)).squeeze(0).contiguous()

    stem = block_file_stem(args.block)
    archive_path = args.output_dir / f"{stem}.elw"
    report_path = args.output_dir / f"{stem}.json"
    manifest = write_archive(
        {
            # Tensor names intentionally mirror the PyTorch block path. The C++
            # test can use these names directly when loading the golden archive.
            f"{args.block}.input": block_input,
            f"{args.block}.expected": block_expected,
        },
        archive_path,
    )
    write_report(
        report_path,
        torch_module,
        torchvision_module,
        weights_label,
        args.block,
        args.seed,
        archive_path,
        manifest,
        block,
    )

    print(f"Model: torchvision ResNet-18 ({weights_label})")
    print(f"Block: {args.block}")
    print("Residual path: identity")
    print(f"Seed: {args.seed}")
    print(f"Input shape: {list(block_input.shape)}")
    print(f"Expected shape: {list(block_expected.shape)}")
    print(f"Conv1 stride: {tuple_of_ints(block.conv1.stride)}")
    print(f"Conv2 stride: {tuple_of_ints(block.conv2.stride)}")
    print(f"Archive: {archive_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
