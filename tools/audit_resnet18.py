#!/usr/bin/env python3
"""Audit and export TorchVision ResNet-18 weights for Eigen-Learn.

The JSON audit is for humans. The `.elw` archive is for the C++ runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elw_archive import ALIGNMENT, write_archive
from resnet18_common import load_resnet18


def module_kind_for_state_key(name: str) -> str:
    suffixes = (
        ".weight",
        ".bias",
        ".running_mean",
        ".running_var",
        ".num_batches_tracked",
    )
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return ""


def audit_modules(model: Any) -> list[dict[str, Any]]:
    import torch.nn as nn

    interesting = (
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.MaxPool2d,
        nn.AdaptiveAvgPool2d,
        nn.Linear,
    )

    modules: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, interesting):
            continue
        entry: dict[str, Any] = {
            "name": name,
            "type": module.__class__.__name__,
        }
        if isinstance(module, nn.Conv2d):
            entry.update(
                {
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": list(module.kernel_size),
                    "stride": list(module.stride),
                    "padding": list(module.padding),
                    "dilation": list(module.dilation),
                    "groups": module.groups,
                    "bias": module.bias is not None,
                }
            )
        elif isinstance(module, nn.BatchNorm2d):
            entry.update(
                {
                    "num_features": module.num_features,
                    "eps": module.eps,
                    "momentum": module.momentum,
                    "affine": module.affine,
                    "track_running_stats": module.track_running_stats,
                }
            )
        elif isinstance(module, nn.Linear):
            entry.update(
                {
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "bias": module.bias is not None,
                }
            )
        elif isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            entry["repr"] = repr(module)
        modules.append(entry)
    return modules


def audit_state_dict(model: Any) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    for name, tensor in model.state_dict().items():
        module_name = module_kind_for_state_key(name)
        module = modules.get(module_name)
        rows.append(
            {
                "name": name,
                "module": module_name,
                "module_type": module.__class__.__name__ if module else "",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
                "nbytes": int(tensor.numel() * tensor.element_size()),
                "element_size": int(tensor.element_size()),
                "requires_grad": False,
                "contiguous": bool(tensor.is_contiguous()),
            }
        )
    return rows


def summarize_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_bytes = sum(row["nbytes"] for row in rows)
    by_dtype: dict[str, dict[str, int]] = {}
    by_module_type: dict[str, dict[str, int]] = {}
    for row in rows:
        dtype_bucket = by_dtype.setdefault(row["dtype"], {"count": 0, "nbytes": 0})
        dtype_bucket["count"] += 1
        dtype_bucket["nbytes"] += row["nbytes"]

        kind = row["module_type"] or "unknown"
        kind_bucket = by_module_type.setdefault(kind, {"count": 0, "nbytes": 0})
        kind_bucket["count"] += 1
        kind_bucket["nbytes"] += row["nbytes"]

    return {
        "tensor_count": len(rows),
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / (1024 * 1024), 3),
        "by_dtype": by_dtype,
        "by_module_type": by_module_type,
    }


def write_audit(
    output_path: Path,
    torch_module: Any,
    torchvision_module: Any,
    weights_label: str,
    model: Any,
    archive_path: Path | None,
    archive_manifest: list[dict[str, Any]],
) -> None:
    state_rows = audit_state_dict(model)
    report = {
        "model": "torchvision.models.resnet18",
        "weights": weights_label,
        "torch_version": torch_module.__version__,
        "torchvision_version": torchvision_module.__version__,
        "state_summary": summarize_state(state_rows),
        "state_dict": state_rows,
        "operation_inventory": audit_modules(model),
        "export": {
            "format": "ELWT0001",
            "archive": str(archive_path) if archive_path else None,
            "alignment_bytes": ALIGNMENT,
            "records": archive_manifest,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit TorchVision ResNet-18 and export Eigen-Learn weights."
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet1k_v1", "default", "none"),
        default="imagenet1k_v1",
        help="TorchVision weight set to load. Use 'none' to avoid downloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/resnet18"),
        help="Directory for audit JSON and weight archive.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Only write the JSON audit; do not write the .elw archive.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch_module, torchvision_module, model, weights_label = load_resnet18(args.weights)

    stem = f"resnet18_{weights_label}"
    audit_path = args.output_dir / "resnet18_audit.json"
    archive_path = args.output_dir / f"{stem}.elw"

    archive_manifest: list[dict[str, Any]] = []
    if not args.no_export:
        archive_manifest = write_archive(model.state_dict(), archive_path)

    write_audit(
        audit_path,
        torch_module,
        torchvision_module,
        weights_label,
        model,
        None if args.no_export else archive_path,
        archive_manifest,
    )

    state_rows = audit_state_dict(model)
    summary = summarize_state(state_rows)
    print(f"Model: torchvision ResNet-18 ({weights_label})")
    print(f"Tensors: {summary['tensor_count']}")
    print(f"Weights: {summary['total_mib']} MiB")
    print(f"Audit: {audit_path}")
    if not args.no_export:
        print(f"Archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
