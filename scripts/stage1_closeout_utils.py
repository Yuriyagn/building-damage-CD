from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_path(data_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(data_root) / path


def to_data_rel(data_root: str | Path, path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(data_root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_manifest(practice_root: str | Path, manifest: str | Path) -> Path:
    path = Path(manifest)
    if path.is_absolute():
        return path
    return Path(practice_root) / path


def default_data_root(practice_root: str | Path) -> Path:
    return Path(practice_root).resolve().parent


def common_stage2_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "event_id": row.get("event_id", ""),
        "canonical_event_name": row.get("canonical_event_name", ""),
        "disaster_type": row.get("disaster_type", ""),
        "country_or_region": row.get("country_or_region", ""),
        "continent": row.get("continent", ""),
        "qc_label": row.get("qc_label", ""),
        "post_modality": "SAR",
    }
