#!/usr/bin/env python3
"""Export a PyTorch golden fixture for ResNet-18 avgpool (head pooling layer)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elw_archive import write_archive
from resnet18_common import load_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deterministic PyTorch input/output for ResNet-18 avgpool."
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
        default=7,
        help="Input feature-map height for the [C, H, W] fixture.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=7,
        help="Input feature-map width for the [C, H, W] fixture.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=512,
        help="Input channel count (512 for ResNet-18 head).",
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
    pool = model.avgpool
    report = {
        "fixture": "resnet18_avgpool_golden",
        "operation": "torchvision.models.resnet18.avgpool",
        "weights": weights_label,
        "torch_version": torch_module.__version__,
        "torchvision_version": torchvision_module.__version__,
        "seed": seed,
        "archive": str(archive_path),
        "records": manifest,
        "avgpool": {
            "output_size": pool.output_size,
            "type": type(pool).__name__,
        },
        "contract": {
            "input": "avgpool.input",
            "expected": "avgpool.expected",
            "meaning": (
                "avgpool.expected = model.avgpool(avgpool.input.unsqueeze(0)).squeeze(0) "
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
        avgpool_input = torch_module.randn(
            args.channels,
            args.height,
            args.width,
            dtype=torch_module.float32,
        )
        avgpool_expected = model.avgpool(
            avgpool_input.unsqueeze(0)
        ).squeeze(0).contiguous()

    archive_path = args.output_dir / "avgpool_golden.elw"
    report_path = args.output_dir / "avgpool_golden.json"
    manifest = write_archive(
        {
            "avgpool.input": avgpool_input,
            "avgpool.expected": avgpool_expected,
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
    print(f"Input shape: {list(avgpool_input.shape)}")
    print(f"Expected shape: {list(avgpool_expected.shape)}")
    print(f"Pool type: {type(model.avgpool).__name__}")
    print(f"Output size: {model.avgpool.output_size}")
    print(f"Archive: {archive_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
