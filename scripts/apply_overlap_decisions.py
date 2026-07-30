#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from overlap_audit_core import json_dump, write_csv


CONFIRMED_LABELS = {"confirmed_overlap", "likely_overlap"}
REVIEWED_LABELS = CONFIRMED_LABELS | {
    "adjacent_no_overlap",
    "same_region_no_overlap",
    "not_overlap",
    "uncertain",
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply reviewed overlap decisions to manifests.")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--policy", choices=("relaxed_train_test_only",), required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--protocol-name", default="", help="Name written to overlap_review_summary.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    candidates_path = args.candidates or args.decisions.parent / "candidates_all.csv"
    candidates = {row["pair_id"]: row for row in read_csv(candidates_path)}
    decisions = {row["pair_id"]: row for row in read_csv(args.decisions) if row.get("pair_id")}
    required = {
        pair_id
        for pair_id, row in candidates.items()
        if row["split_pair"] == "train-test" and row["risk_level"] in {"high", "medium"}
    }
    incomplete = sorted(
        pair_id
        for pair_id in required
        if pair_id not in decisions or decisions[pair_id].get("review_label") not in REVIEWED_LABELS
    )
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(
            f"{len(incomplete)} required train-test high/medium candidates are unreviewed; "
            "complete review or pass --allow-incomplete only for a non-final diagnostic"
        )

    remove_ids: set[str] = set()
    removed_rows: list[dict[str, str]] = []
    clusters = UnionFind()
    for pair_id, decision in decisions.items():
        if pair_id not in candidates:
            raise KeyError(f"decision references unknown pair_id: {pair_id}")
        candidate = candidates[pair_id]
        label = decision.get("review_label", "")
        action = decision.get("review_action", "")
        if label not in CONFIRMED_LABELS:
            continue
        clusters.union(candidate["id_a"], candidate["id_b"])
        if candidate["split_pair"] != "train-test":
            continue
        remove_id = ""
        remove_split = ""
        if action == "exclude_train_side":
            side = "a" if candidate["split_a"] == "train" else "b"
            remove_id, remove_split = candidate[f"id_{side}"], "train"
        elif action == "exclude_test_side":
            side = "a" if candidate["split_a"] == "test" else "b"
            remove_id, remove_split = candidate[f"id_{side}"], "test"
        elif action in {"keep_both_record_only", "cluster_same_area", "needs_second_review"}:
            continue
        else:
            raise ValueError(f"invalid action for confirmed train-test pair {pair_id}: {action}")
        remove_ids.add(remove_id)
        removed_rows.append(
            {
                "pair_id": pair_id,
                "removed_id": remove_id,
                "removed_split": remove_split,
                "review_label": label,
                "review_action": action,
                "review_comment": decision.get("review_comment", ""),
            }
        )

    cluster_members: dict[str, list[str]] = {}
    for sample_id in clusters.parent:
        cluster_members.setdefault(clusters.find(sample_id), []).append(sample_id)
    cluster_ids: dict[str, str] = {}
    for index, members in enumerate(sorted(cluster_members.values(), key=lambda values: sorted(values)), 1):
        name = f"overlap_cluster_{index:05d}"
        for sample_id in members:
            cluster_ids[sample_id] = name

    manifest_root = args.manifest_root.resolve()
    manifest_files = sorted(manifest_root.glob("stage2_master_*.jsonl"))
    manifest_files += sorted(manifest_root.glob("*/*.jsonl"))
    if not manifest_files:
        raise FileNotFoundError(f"no manifests in {manifest_root}")
    counts_before: dict[str, int] = {}
    counts_after: dict[str, int] = {}
    output_rows: dict[Path, list[dict[str, object]]] = {}
    for path in manifest_files:
        relative = path.relative_to(manifest_root)
        rows = read_jsonl(path)
        manifest_key = str(relative)
        counts_before[manifest_key] = len(rows)
        kept: list[dict[str, object]] = []
        for row in rows:
            sample_id = str(row["id"])
            if sample_id in remove_ids:
                continue
            if sample_id in cluster_ids:
                row["overlap_cluster_id"] = cluster_ids[sample_id]
            kept.append(row)
        counts_after[manifest_key] = len(kept)
        output_rows[relative] = kept

    summary = {
        "protocol": args.protocol_name or f"{manifest_root.name}-overlap-reviewed",
        "policy": args.policy,
        "source_manifest_root": str(manifest_root),
        "candidates": str(candidates_path.resolve()),
        "decisions": str(args.decisions.resolve()),
        "required_train_test_reviews": len(required),
        "incomplete_required_reviews": len(incomplete),
        "decision_count": len(decisions),
        "removed_unique_sample_count": len(remove_ids),
        "overlap_cluster_count": len(cluster_members),
        "counts_before": dict(counts_before),
        "counts_after": dict(counts_after),
        "final": not incomplete,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    for relative, rows in output_rows.items():
        write_jsonl(args.out_root / relative, rows)
    source_plan = manifest_root / "split_plan_resolved.json"
    if source_plan.is_file():
        shutil.copy2(source_plan, args.out_root / "source_split_plan_resolved.json")
    write_csv(
        args.out_root / "removed_samples.csv",
        removed_rows,
        ["pair_id", "removed_id", "removed_split", "review_label", "review_action", "review_comment"],
    )
    json_dump(args.out_root / "overlap_review_summary.json", summary)
    print(f"[OK] wrote {args.out_root} removed={len(remove_ids)} final={summary['final']}")


if __name__ == "__main__":
    main()
