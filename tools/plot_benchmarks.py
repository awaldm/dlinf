#!/usr/bin/env python3
"""Generate simple SVG benchmark charts from bench_kernels JSONL output."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                msg = f"{path}:{line_number}: invalid JSONL record: {exc}"
                raise SystemExit(msg) from exc
    if not records:
        raise SystemExit(f"{path}: no benchmark records found")
    return records


def label_for(record: dict[str, Any]) -> str:
    op = str(record["op"])
    impl = str(record["impl"])
    labels = {
        ("linear_fc", "linear_naive"): "Linear / direct",
        ("linear_fc", "linear_eigen"): "Linear / Eigen",
        ("conv1", "conv2d_naive_direct"): "Conv1 / direct",
        ("conv1", "conv2d_im2col_eigen"): "Conv1 / im2col+Eigen",
        (
            "conv1_bn1",
            "conv2d_naive_direct_batchnorm2d_direct",
        ): "Conv1+BN / direct",
        (
            "conv1_bn1",
            "conv2d_im2col_eigen_batchnorm2d_direct",
        ): "Conv1+BN / im2col+Eigen",
    }
    return labels.get((op, impl), f"{op} / {impl}")


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        "<style>"
        "text{font-family:Inter,Arial,sans-serif;fill:#1f2937}"
        ".title{font-size:18px;font-weight:700}"
        ".subtitle{font-size:12px;fill:#4b5563}"
        ".label{font-size:12px}"
        ".value{font-size:12px;font-weight:600}"
        ".axis{stroke:#d1d5db;stroke-width:1}"
        ".bar-direct{fill:#64748b}"
        ".bar-eigen{fill:#2563eb}"
        ".bar-speed{fill:#059669}"
        "</style>"
    )


def write_latency_svg(records: list[dict[str, Any]], path: Path) -> None:
    width = 980
    row_height = 40
    top = 84
    left = 245
    bar_width = 545
    height = top + len(records) * row_height + 48

    values = [float(record["median_ms"]) for record in records]
    min_value = max(min(values) * 0.8, 0.001)
    max_value = max(values) * 1.15
    log_span = math.log10(max_value) - math.log10(min_value)

    parts = [svg_header(width, height)]
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append('<text class="title" x="28" y="34">Kernel latency</text>')
    parts.append(
        '<text class="subtitle" x="28" y="56">'
        "Median latency in milliseconds. Bar length uses a log scale."
        "</text>"
    )
    parts.append(f'<line class="axis" x1="{left}" y1="{top - 12}" x2="{left + bar_width}" y2="{top - 12}"/>')

    for index, record in enumerate(records):
        y = top + index * row_height
        value = float(record["median_ms"])
        fraction = (math.log10(value) - math.log10(min_value)) / log_span if log_span else 1.0
        length = max(3.0, fraction * bar_width)
        impl = str(record["impl"])
        css_class = "bar-eigen" if "eigen" in impl else "bar-direct"
        label = html.escape(label_for(record))
        parts.append(f'<text class="label" x="28" y="{y + 18}">{label}</text>')
        parts.append(f'<rect class="{css_class}" x="{left}" y="{y}" width="{length:.1f}" height="22" rx="3"/>')
        parts.append(f'<text class="value" x="{left + length + 10:.1f}" y="{y + 16}">{value:.3f} ms</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def find_record(records: list[dict[str, Any]], op: str, impl: str) -> dict[str, Any]:
    for record in records:
        if record.get("op") == op and record.get("impl") == impl:
            return record
    raise SystemExit(f"missing benchmark record for {op} / {impl}")


def write_speedup_svg(records: list[dict[str, Any]], path: Path) -> None:
    pairs = [
        (
            "Linear",
            find_record(records, "linear_fc", "linear_naive"),
            find_record(records, "linear_fc", "linear_eigen"),
        ),
        (
            "Conv1",
            find_record(records, "conv1", "conv2d_naive_direct"),
            find_record(records, "conv1", "conv2d_im2col_eigen"),
        ),
        (
            "Conv1+BN",
            find_record(records, "conv1_bn1", "conv2d_naive_direct_batchnorm2d_direct"),
            find_record(records, "conv1_bn1", "conv2d_im2col_eigen_batchnorm2d_direct"),
        ),
    ]

    speedups = [
        float(direct["median_ms"]) / float(eigen["median_ms"])
        for _, direct, eigen in pairs
    ]
    max_speedup = max(speedups) * 1.2

    width = 820
    row_height = 46
    top = 84
    left = 155
    bar_width = 510
    height = top + len(pairs) * row_height + 48

    parts = [svg_header(width, height)]
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append('<text class="title" x="28" y="34">Eigen-backed speedup</text>')
    parts.append(
        '<text class="subtitle" x="28" y="56">'
        "Median direct latency divided by median Eigen-backed latency."
        "</text>"
    )

    for index, ((name, _direct, _eigen), speedup) in enumerate(zip(pairs, speedups)):
        y = top + index * row_height
        length = (speedup / max_speedup) * bar_width
        parts.append(f'<text class="label" x="28" y="{y + 18}">{html.escape(name)}</text>')
        parts.append(f'<rect class="bar-speed" x="{left}" y="{y}" width="{length:.1f}" height="24" rx="3"/>')
        parts.append(f'<text class="value" x="{left + length + 10:.1f}" y="{y + 17}">{speedup:.1f}x</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--latency-output", required=True, type=Path)
    parser.add_argument("--speedup-output", required=True, type=Path)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    args.latency_output.parent.mkdir(parents=True, exist_ok=True)
    args.speedup_output.parent.mkdir(parents=True, exist_ok=True)
    write_latency_svg(records, args.latency_output)
    write_speedup_svg(records, args.speedup_output)


if __name__ == "__main__":
    main()
