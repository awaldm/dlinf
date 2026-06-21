#!/usr/bin/env python3
"""Export a PyTorch golden fixture for the ResNet-18 first Conv2d layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elw_archive import write_archive
from resnet18_common import load_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deterministic PyTorch input/output for ResNet-18 model.conv1."
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet1k_v1", "default", "none"),
        default="imagenet1k_v1",
        help="TorchVision weight set to load.",
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
        help="Torch RNG seed used for the deterministic conv1.input tensor.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Input image height for the [3, H, W] fixture.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Input image width for the [3, H, W] fixture.",
    )
    return parser.parse_args()


def tuple_of_ints(values: Any) -> list[int]:
    return [int(value) for value in values]


def write_report(
    output_path: Path,
    torch_module: Any,
    torchvision_module: Any,
    weights_label: str,
    seed: int,
    archive_path: Path,
    manifest: list[dict[str, Any]],
    model: Any,
) -> None:
    conv1 = model.conv1
    report = {
        "fixture": "resnet18_conv1_golden",
        "operation": "torchvision.models.resnet18.conv1",
        "weights": weights_label,
        "torch_version": torch_module.__version__,
        "torchvision_version": torchvision_module.__version__,
        "seed": seed,
        "archive": str(archive_path),
        "records": manifest,
        "conv2d": {
            "weight": "conv1.weight",
            "bias": None,
            "in_channels": int(conv1.in_channels),
            "out_channels": int(conv1.out_channels),
            "kernel_size": tuple_of_ints(conv1.kernel_size),
            "stride": tuple_of_ints(conv1.stride),
            "padding": tuple_of_ints(conv1.padding),
            "dilation": tuple_of_ints(conv1.dilation),
            "groups": int(conv1.groups),
        },
        "contract": {
            "input": "conv1.input",
            "expected": "conv1.expected",
            "meaning": "conv1.expected = model.conv1(conv1.input.unsqueeze(0)).squeeze(0) computed by PyTorch",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise SystemExit("--height and --width must be positive")

    torch_module, torchvision_module, model, weights_label = load_resnet18(args.weights)

    torch_module.manual_seed(args.seed)

    with torch_module.inference_mode():
        conv_input = torch_module.randn(
            3,
            args.height,
            args.width,
            dtype=torch_module.float32,
        )
        conv_expected = model.conv1(conv_input.unsqueeze(0)).squeeze(0).contiguous()

    archive_path = args.output_dir / "conv1_golden.elw"
    report_path = args.output_dir / "conv1_golden.json"
    manifest = write_archive(
        {
            "conv1.input": conv_input,
            "conv1.expected": conv_expected,
        },
        archive_path,
    )
    write_report(
        report_path,
        torch_module,
        torchvision_module,
        weights_label,
        args.seed,
        archive_path,
        manifest,
        model,
    )

    print(f"Model: torchvision ResNet-18 ({weights_label})")
    print(f"Seed: {args.seed}")
    print(f"Input shape: {list(conv_input.shape)}")
    print(f"Expected shape: {list(conv_expected.shape)}")
    print(f"Stride: {tuple_of_ints(model.conv1.stride)}")
    print(f"Padding: {tuple_of_ints(model.conv1.padding)}")
    print(f"Archive: {archive_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
