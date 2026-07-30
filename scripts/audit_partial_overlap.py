#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from overlap_audit_core import (
    IMAGE_FIELDS,
    PAIR_ORDER,
    FeatureCache,
    LocalFeatureCache,
    generate_candidate_indices,
    json_dump,
    load_samples,
    local_descriptor_candidate_pairs,
    risk_counts,
    score_candidate,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect exact-near-partial image overlap candidates.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--protocol", required=True, choices=("legacy", "strict_v1"))
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--global-top-k", type=int, default=20)
    parser.add_argument("--minimum-score", type=float, default=0.60)
    parser.add_argument("--include-within-split", action="store_true")
    parser.add_argument("--disable-orb-retrieval", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve()
    target = out_root / "candidates_all.csv"
    if target.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing audit: {target}; pass --force explicitly")
    started = time.time()
    data_root = args.data_root.resolve()
    manifest_root = args.manifest_root.resolve()
    samples = load_samples(data_root, manifest_root)
    feature_cache = FeatureCache(args.feature_cache.resolve())

    features: dict[str, dict[str, list]] = {}
    unique_paths = {
        sample.path_for(field)
        for split_samples in samples.values()
        for sample in split_samples
        for field in IMAGE_FIELDS
    }
    print(f"[INFO] protocol={args.protocol} samples={sum(map(len, samples.values()))} unique_paths={len(unique_paths)}")
    for split, split_samples in samples.items():
        features[split] = {field: [] for field in IMAGE_FIELDS}
        for sample in tqdm(split_samples, desc=f"features_{split}"):
            for field in IMAGE_FIELDS:
                features[split][field].append(feature_cache.get(sample.path_for(field)))

    local_cache = LocalFeatureCache()
    rows: list[dict[str, object]] = []
    pair_summaries: dict[str, object] = {}
    csv_fields: list[str] | None = None
    pair_order = list(PAIR_ORDER)
    if args.include_within_split:
        pair_order.extend((split, split) for split in ("train", "val", "test"))
    for split_a, split_b in pair_order:
        candidate_indices = generate_candidate_indices(
            features[split_a], features[split_b], global_top_k=args.global_top_k
        )
        local_candidates = (
            set()
            if args.disable_orb_retrieval
            else local_descriptor_candidate_pairs(samples[split_a], samples[split_b], local_cache)
        )
        candidate_indices |= local_candidates
        if split_a == split_b:
            candidate_indices = {(left, right) for left, right in candidate_indices if left < right}
        print(f"[INFO] {split_a}-{split_b} generated_candidates={len(candidate_indices)}")
        pair_rows: list[dict[str, object]] = []
        for index_a, index_b in tqdm(sorted(candidate_indices), desc=f"score_{split_a}_{split_b}"):
            field_a = {field: features[split_a][field][index_a] for field in IMAGE_FIELDS}
            field_b = {field: features[split_b][field][index_b] for field in IMAGE_FIELDS}
            row = score_candidate(
                samples[split_a][index_a], samples[split_b][index_b], field_a, field_b, local_cache
            )
            if float(row["overlap_score"]) >= args.minimum_score:
                pair_rows.append(row)
        pair_rows.sort(key=lambda row: (-float(row["overlap_score"]), str(row["pair_id"])))
        if pair_rows and csv_fields is None:
            csv_fields = list(pair_rows[0])
        pair_name = f"{split_a}_{split_b}"
        if csv_fields is None:
            raise RuntimeError("no candidates survived; lower --minimum-score to retain audit rows")
        write_csv(out_root / f"candidates_{pair_name}.csv", pair_rows, csv_fields)
        rows.extend(pair_rows)
        pair_summaries[f"{split_a}-{split_b}"] = {
            "generated_candidate_count": len(candidate_indices),
            "local_descriptor_candidate_count": len(local_candidates),
            "retained_candidate_count": len(pair_rows),
            "risk_counts": risk_counts(pair_rows),
        }

    rows.sort(key=lambda row: (-float(row["overlap_score"]), str(row["pair_id"])))
    assert csv_fields is not None
    write_csv(target, rows, csv_fields)
    write_csv(out_root / "review_decisions.csv", [], ["pair_id", "review_label", "review_action", "review_comment", "reviewed_at"])
    summary = {
        "protocol": args.protocol,
        "policy": "relaxed_train_test_only",
        "data_root": str(data_root),
        "manifest_root": str(manifest_root),
        "sample_counts": {split: len(value) for split, value in samples.items()},
        "global_top_k": args.global_top_k,
        "minimum_score": args.minimum_score,
        "pairs": pair_summaries,
        "all_retained_candidates": len(rows),
        "risk_counts": risk_counts(rows),
        "elapsed_seconds": round(time.time() - started, 3),
        "limitations": [
            "Candidate retrieval is image-based because the PNG package has no geospatial footprints.",
            "Absence of a candidate is not a formal proof of spatial disjointness.",
            "All high/medium train-test candidates require human review before manifest changes.",
        ],
    }
    json_dump(out_root / "audit_summary.json", summary)
    print(f"[OK] wrote {target} retained={len(rows)} elapsed={summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
