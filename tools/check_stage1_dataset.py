#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import zlib
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve(data_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return data_root / path


def read_png_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        sig = handle.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"not a PNG: {path}")
        length = struct.unpack(">I", handle.read(4))[0]
        kind = handle.read(4)
        if kind != b"IHDR" or length != 13:
            raise ValueError(f"missing IHDR: {path}")
        data = handle.read(13)
    return struct.unpack(">I", data[0:4])[0], struct.unpack(">I", data[4:8])[0]


def gray_png_values(path: Path) -> set[int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    width = height = bit_depth = color_type = interlace = None
    parts = []
    pos = 8
    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width = struct.unpack(">I", payload[0:4])[0]
            height = struct.unpack(">I", payload[4:8])[0]
            bit_depth = payload[8]
            color_type = payload[9]
            interlace = payload[12]
        elif kind == b"IDAT":
            parts.append(payload)
        elif kind == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type != 0 or interlace != 0:
        raise ValueError(f"unsupported gray PNG: {path}")
    raw = zlib.decompress(b"".join(parts))
    stride = int(width)
    values = set()
    for y in range(int(height)):
        offset = y * (stride + 1)
        filter_type = raw[offset]
        if filter_type != 0:
            raise ValueError(f"unsupported PNG filter {filter_type}: {path}")
        values.update(raw[offset + 1: offset + 1 + stride])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--check-mask-values", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        candidate = data_root / manifest
        if not candidate.exists():
            candidate = data_root / "manifests" / manifest.name
        manifest = candidate

    rows = read_jsonl(manifest)
    missing = []
    bad_masks = []
    dims = Counter()
    disasters = Counter()
    for row in rows:
        image = resolve(data_root, row["image"])
        mask = resolve(data_root, row["mask_binary"])
        if not image.exists():
            missing.append(str(image))
            continue
        if not mask.exists():
            missing.append(str(mask))
            continue
        disasters[str(row.get("disaster_type", "") or "unknown")] += 1
        dims[read_png_header(image)] += 1
        if args.check_mask_values:
            values = gray_png_values(mask)
            if values - {0, 1}:
                bad_masks.append({"id": row.get("id", ""), "values": sorted(values), "mask": str(mask)})

    out = {
        "manifest": str(manifest),
        "row_count": len(rows),
        "missing_count": len(missing),
        "bad_mask_count": len(bad_masks),
        "image_dims": {f"{w}x{h}": n for (w, h), n in sorted(dims.items())},
        "disaster_counts": dict(sorted(disasters.items())),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    if missing:
        print("First missing paths:")
        for item in missing[:20]:
            print(item)
        raise SystemExit(1)
    if bad_masks:
        print("First bad masks:")
        for item in bad_masks[:20]:
            print(item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
