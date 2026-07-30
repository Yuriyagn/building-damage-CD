from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CLASS_NAMES = {
    0: "background",
    1: "intact",
    2: "damaged",
    3: "destroyed",
}


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_manifest(data_root: str | Path, manifest: str | Path) -> Path:
    manifest_path = Path(manifest)
    if manifest_path.is_absolute() and manifest_path.exists():
        return manifest_path
    if not manifest_path.is_absolute() and manifest_path.exists():
        return manifest_path.resolve()
    root = Path(data_root)
    candidates = [
        root / manifest_path,
        root / "manifests" / manifest_path,
        root / "manifests" / manifest_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"manifest not found: {manifest}; tried {candidates}")


def resolve_data_path(data_root: str | Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(data_root) / path


def get_metadata(row: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(row.get("id", "")),
        "split": str(row.get("split", "")),
        "event_id": str(row.get("event_id", "")),
        "disaster_type": str(row.get("disaster_type", "")),
        "country_or_region": str(row.get("country_or_region", "")),
        "region": str(row.get("region", "")),
        "qc_label": str(row.get("qc_label", "")),
        "prior_type": str(row.get("prior_type", "")),
    }
