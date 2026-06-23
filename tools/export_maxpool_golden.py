#!/usr/bin/env python3
"""Export a PyTorch golden fixture for ResNet-18 maxpool (stem pooling layer)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elw_archive import write_archive
from resnet18_common import load_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deterministic PyTorch input/output for ResNet-18 maxpool."
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
        help="Torch RNG seed used for the deterministic input tensor.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=112,
        help="Input feature-map height for the [C, H, W] fixture.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=112,
        help="Input feature-map width for the [C, H, W] fixture.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=64,
        help="Input channel count (64 for ResNet-18 stem).",
    )
    return parser.parse_args()


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
    pool = model.maxpool
    report = {
        "fixture": "resnet18_maxpool_golden",
        "operation": "torchvision.models.resnet18.maxpool",
        "weights": weights_label,
        "torch_version": torch_module.__version__,
        "torchvision_version": torchvision_module.__version__,
        "seed": seed,
        "archive": str(archive_path),
        "records": manifest,
        "maxpool": {
            "kernel_size": int(pool.kernel_size)
            if isinstance(pool.kernel_size, int)
            else list(pool.kernel_size),
            "stride": int(pool.stride) if isinstance(pool.stride, int) else list(pool.stride),
            "padding": int(pool.padding) if isinstance(pool.padding, int) else list(pool.padding),
        },
        "contract": {
            "input": "maxpool.input",
            "expected": "maxpool.expected",
            "meaning": (
                "maxpool.expected = model.maxpool(maxpool.input.unsqueeze(0)).squeeze(0) "
                "computed by PyTorch"
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

    torch_module.manual_seed(args.seed)

    with torch_module.inference_mode():
        maxpool_input = torch_module.randn(
            args.channels,
            args.height,
            args.width,
            dtype=torch_module.float32,
        )
        maxpool_expected = model.maxpool(
            maxpool_input.unsqueeze(0)
        ).squeeze(0).contiguous()

    archive_path = args.output_dir / "maxpool_golden.elw"
    report_path = args.output_dir / "maxpool_golden.json"
    manifest = write_archive(
        {
            "maxpool.input": maxpool_input,
            "maxpool.expected": maxpool_expected,
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
    print(f"Input shape: {list(maxpool_input.shape)}")
    print(f"Expected shape: {list(maxpool_expected.shape)}")
    print(f"Kernel: {model.maxpool.kernel_size}")
    print(f"Stride: {model.maxpool.stride}")
    print(f"Padding: {model.maxpool.padding}")
    print(f"Archive: {archive_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
