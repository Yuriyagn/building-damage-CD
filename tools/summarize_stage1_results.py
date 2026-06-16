#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    root = Path(args.outputs_root)
    for path in sorted(root.glob("*/test*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(root)
        rows.append({
            "run": rel.parts[0],
            "eval": rel.parts[1],
            **metrics,
        })
    write_csv(Path(args.out), rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
