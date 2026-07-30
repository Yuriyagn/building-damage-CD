#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize relaxed-overlap-v1 machine and review status.")
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    lines = ["# relaxed-overlap-v1 audit summary", ""]
    validation = args.audit_root / "validation/synthetic_validation.json"
    if validation.is_file():
        value = json.loads(validation.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Synthetic validation",
                "",
                f"- recall@20: `{value['recall_at_20']:.4f}`",
                f"- high-risk recall: `{value['high_risk_recall']:.4f}`",
                f"- adjacent negative medium/high rate: `{value['adjacent_negative_medium_or_high_rate']:.4f}`",
                "",
            ]
        )
    for protocol in ("legacy", "strict_v1"):
        root = args.audit_root / protocol
        candidates = read_csv(root / "candidates_all.csv")
        decisions = {row["pair_id"]: row for row in read_csv(root / "review_decisions.csv") if row.get("pair_id")}
        lines.extend([f"## {protocol}", ""])
        if not candidates:
            lines.extend(["Automatic audit pending or produced no retained candidates.", ""])
            continue
        lines.extend(
            [
                "| Pair | High | Medium | Low | Reviewed |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for pair in sorted({row["split_pair"] for row in candidates}):
            selected = [row for row in candidates if row["split_pair"] == pair]
            risks = Counter(row["risk_level"] for row in selected)
            reviewed = sum(row["pair_id"] in decisions for row in selected)
            lines.append(
                f"| {pair} | {risks['high']} | {risks['medium']} | {risks['low']} | {reviewed} |"
            )
        required = [
            row
            for row in candidates
            if row["split_pair"] == "train-test" and row["risk_level"] in {"high", "medium"}
        ]
        completed = sum(
            row["pair_id"] in decisions and decisions[row["pair_id"]].get("review_label")
            for row in required
        )
        lines.extend(
            [
                "",
                f"Required train-test high/medium review: `{completed}/{len(required)}`.",
                "",
            ]
        )
    out = args.out or args.audit_root / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
