"""Shared dlinf weight archive writer.

The `.elw` format is intentionally simple: fixed-size metadata records followed
by aligned raw tensor payloads. Keep all Python writers on this module so the
binary format does not drift between audit and golden-test exports.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


MAGIC = b"ELWT0001"
VERSION = 1
ENDIAN_TAG = 0x01020304
HEADER_STRUCT = struct.Struct("<8sIIIIQQQQQ")
RECORD_STRUCT = struct.Struct("<128sIIQQII8Q")
HEADER_BYTES = HEADER_STRUCT.size
RECORD_BYTES = RECORD_STRUCT.size
ALIGNMENT = 64

DTYPE_CODES = {
    "torch.float32": 1,
    "torch.float64": 2,
    "torch.int64": 3,
    "torch.int32": 4,
    "torch.uint8": 5,
    "torch.float16": 6,
}


@dataclass(frozen=True)
class TensorExport:
    name: str
    dtype_name: str
    dtype_code: int
    shape: tuple[int, ...]
    elem_size: int
    payload: bytes

    @property
    def nbytes(self) -> int:
        return len(self.payload)


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def iter_named_tensors(
    tensors: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    if isinstance(tensors, Mapping):
        return list(tensors.items())
    return list(tensors)


def tensor_to_export(name: str, tensor: Any) -> TensorExport:
    dtype_name = str(tensor.dtype)
    dtype_code = DTYPE_CODES.get(dtype_name)
    if dtype_code is None:
        raise ValueError(f"Unsupported dtype for export: {name} has {dtype_name}")
    if len(tensor.shape) > 8:
        raise ValueError(
            f"Unsupported rank for export: {name} has rank {len(tensor.shape)}"
        )
    if len(name.encode("utf-8")) >= 128:
        raise ValueError(f"Tensor name too long for fixed record: {name}")

    cpu_tensor = tensor.detach().cpu().contiguous()
    array = cpu_tensor.numpy()
    if sys.byteorder != "little":
        array = array.byteswap()
    payload = array.tobytes(order="C")
    return TensorExport(
        name=name,
        dtype_name=dtype_name,
        dtype_code=dtype_code,
        shape=tuple(int(dim) for dim in cpu_tensor.shape),
        elem_size=int(cpu_tensor.element_size()),
        payload=payload,
    )


def write_archive(
    tensors: Mapping[str, Any] | Iterable[tuple[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    exports = [
        tensor_to_export(name, tensor) for name, tensor in iter_named_tensors(tensors)
    ]

    records_offset = HEADER_BYTES
    data_offset = align_up(records_offset + len(exports) * RECORD_BYTES)
    cursor = data_offset

    packed_records: list[bytes] = []
    payload_chunks: list[tuple[int, bytes]] = []
    manifest_rows: list[dict[str, Any]] = []

    for item in exports:
        cursor = align_up(cursor)
        shape = list(item.shape) + [0] * (8 - len(item.shape))
        packed_records.append(
            RECORD_STRUCT.pack(
                item.name.encode("utf-8"),
                item.dtype_code,
                len(item.shape),
                cursor,
                item.nbytes,
                item.elem_size,
                1,
                *shape,
            )
        )
        payload_chunks.append((cursor, item.payload))
        manifest_rows.append(
            {
                "name": item.name,
                "dtype": item.dtype_name,
                "dtype_code": item.dtype_code,
                "shape": list(item.shape),
                "offset": cursor,
                "nbytes": item.nbytes,
            }
        )
        cursor += item.nbytes

    file_bytes = cursor
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        ENDIAN_TAG,
        HEADER_BYTES,
        RECORD_BYTES,
        len(exports),
        records_offset,
        data_offset,
        file_bytes,
        0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(header)
        for record in packed_records:
            handle.write(record)
        handle.write(b"\x00" * (data_offset - handle.tell()))
        for offset, payload in payload_chunks:
            handle.write(b"\x00" * (offset - handle.tell()))
            handle.write(payload)

    return manifest_rows
