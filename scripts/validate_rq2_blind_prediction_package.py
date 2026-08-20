#!/usr/bin/env python3
"""Validate the frozen RQ2 prediction archive without reading hidden labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_rq2_blind_derangement import read_blind_manifest, sha256_file  # noqa: E402


METHODS = ("F0", "candidate")
CONDITIONS = ("C2", "C3")
SEEDS = (42, 3407, 2026)


def validate_package(manifest: Path, package_root: Path) -> dict[str, Any]:
    rows = read_blind_manifest(manifest)
    expected = {str(row["id"]): (int(row["width"]), int(row["height"])) for row in rows}
    files = []
    errors = []
    for method in METHODS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                directory = package_root / "predictions" / method / condition / f"seed_{seed}"
                observed = {path.stem: path for path in directory.glob("*.png")} if directory.is_dir() else {}
                missing = sorted(set(expected) - set(observed))
                extra = sorted(set(observed) - set(expected))
                if missing or extra:
                    errors.append(
                        f"{method}/{condition}/seed_{seed}: missing={missing[:5]}, extra={extra[:5]}"
                    )
                    continue
                for sample_id in sorted(expected):
                    path = observed[sample_id]
                    array = np.asarray(Image.open(path))
                    width, height = expected[sample_id]
                    if array.ndim != 2 or array.shape != (height, width):
                        errors.append(
                            f"{path}: expected {(height, width)} single-channel, got {array.shape}"
                        )
                        continue
                    if array.dtype != np.uint8:
                        errors.append(f"{path}: expected uint8 PNG, got {array.dtype}")
                        continue
                    unique = set(int(value) for value in np.unique(array))
                    if not unique <= {0, 1, 2, 3}:
                        errors.append(f"{path}: invalid grade values {sorted(unique)}")
                        continue
                    files.append(
                        {
                            "path": str(path.relative_to(package_root)),
                            "sha256": sha256_file(path),
                            "width": width,
                            "height": height,
                        }
                    )
    if errors:
        raise ValueError("blind prediction package rejected:\n- " + "\n- ".join(errors))
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(item["path"].encode("utf-8"))
        aggregate.update(item["sha256"].encode("ascii"))
    return {
        "protocol_id": "rq2_model_development_v1.0",
        "status": "format_valid_no_scores_computed",
        "manifest_sha256": sha256_file(manifest),
        "sample_count": len(rows),
        "prediction_count": len(files),
        "prediction_tree_sha256": aggregate.hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()
    audit = validate_package(args.manifest.resolve(), args.package_root.resolve())
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": audit["status"], "prediction_count": audit["prediction_count"]}))


if __name__ == "__main__":
    main()
