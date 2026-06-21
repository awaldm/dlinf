#!/usr/bin/env python3
"""Export a PyTorch golden fixture for the ResNet-18 final Linear layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elw_archive import write_archive
from resnet18_common import load_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deterministic PyTorch input/output for ResNet-18 model.fc."
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
        help="Torch RNG seed used for the deterministic fc.input vector.",
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
) -> None:
    report = {
        "fixture": "resnet18_fc_golden",
        "operation": "torchvision.models.resnet18.fc",
        "weights": weights_label,
        "torch_version": torch_module.__version__,
        "torchvision_version": torchvision_module.__version__,
        "seed": seed,
        "archive": str(archive_path),
        "records": manifest,
        "contract": {
            "input": "fc.input",
            "expected": "fc.expected",
            "meaning": "fc.expected = model.fc(fc.input) computed by PyTorch",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    # Get the resnet18
    torch_module, torchvision_module, model, weights_label = load_resnet18(args.weights)

    # seed
    torch_module.manual_seed(args.seed)

    #
    in_features = int(model.fc.in_features)
    print("number of input features: " + str(in_features))

    # Run inference (dont track gradients, do do any autograd graph building stuff)
    with torch_module.inference_mode():
        fc_input = torch_module.randn(in_features, dtype=torch_module.float32)
        fc_expected = model.fc(fc_input)

    archive_path = args.output_dir / "fc_golden.elw"
    report_path = args.output_dir / "fc_golden.json"
    manifest = write_archive(
        {
            "fc.input": fc_input,
            "fc.expected": fc_expected,
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
    )

    print(f"Model: torchvision ResNet-18 ({weights_label})")
    print(f"Seed: {args.seed}")
    print(f"Input shape: {list(fc_input.shape)}")
    print(f"Expected shape: {list(fc_expected.shape)}")
    print(f"Archive: {archive_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
