#!/usr/bin/env python3
"""Export a PyTorch golden fixture for one ResNet-18 projection BasicBlock.

A projection block differs from an identity block in that the skip path
contains a 1x1 convolution plus BatchNorm to match channel count and spatial
size. The canonical example is ``layer2.0``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elw_archive import write_archive
from resnet18_common import load_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export deterministic PyTorch input/output for a ResNet-18 "
            "projection BasicBlock."
        ),
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet1k_v1", "default", "none"),
        default="imagenet1k_v1",
        help="TorchVision weight set to load.",
    )
    parser.add_argument(
        "--block",
        default="layer2.0",
        help="Dotted ResNet-18 projection BasicBlock path.",
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
    return [int(value) for value in values]


def block_file_stem(block_path: str) -> str:
    return f"{block_path.replace('.', '_')}_projection_basicblock_golden"


def require_projection_basicblock(block_path: str, block: Any) -> None:
    required_attrs = ("conv1", "bn1", "relu", "conv2", "bn2")
    missing = [name for name in required_attrs if not hasattr(block, name)]
    if missing:
        raise SystemExit(
            f"{block_path} is not a supported ResNet BasicBlock; missing {missing}"
        )
    if getattr(block, "downsample", None) is None:
        raise SystemExit(
            f"{block_path} has no downsample/projection residual path. "
            "Use export_basicblock_golden.py for identity blocks."
        )


def conv2d_report(block_path: str, name: str, conv: Any) -> dict[str, Any]:
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
    stem = block_file_stem(block_path)
    downsample = block.downsample
    downsample_conv = downsample[0]
    downsample_bn = downsample[1]

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
            "residual_path": "projection",
            "conv1": conv2d_report(block_path, "conv1", block.conv1),
            "bn1": batchnorm2d_report(block_path, "bn1", block.bn1),
            "activation": "relu",
            "conv2": conv2d_report(block_path, "conv2", block.conv2),
            "bn2": batchnorm2d_report(block_path, "bn2", block.bn2),
            "downsample": {
                "conv": conv2d_report(block_path, "downsample.0", downsample_conv),
                "bn": batchnorm2d_report(block_path, "downsample.1", downsample_bn),
            },
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

    require_projection_basicblock(args.block, block)

    torch_module.manual_seed(args.seed)
    in_channels = int(block.conv1.in_channels)

    with torch_module.inference_mode():
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
    print("Residual path: projection")
    print(f"Seed: {args.seed}")
    print(f"Input shape: {list(block_input.shape)}")
    print(f"Expected shape: {list(block_expected.shape)}")
    print(f"Conv1 stride: {tuple_of_ints(block.conv1.stride)}")
    print(f"Conv2 stride: {tuple_of_ints(block.conv2.stride)}")
    print(f"Downsample conv: {tuple_of_ints(block.downsample[0].kernel_size)}, "
          f"stride={tuple_of_ints(block.downsample[0].stride)}")
    print(f"Archive: {archive_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
