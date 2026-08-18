#!/usr/bin/env python3
"""Attach event-excluded Stage-1 probability priors to every RQ1 fold role."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def key_for(excluded: set[str] | frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(excluded))


def attach(row: dict[str, Any], prior: dict[str, Any], model: dict[str, Any], outer: int, role: str) -> dict[str, Any]:
    output = dict(row)
    output.update({
        "building_prior": prior["building_prob"],
        "building_prior_binary": prior["building_binary"],
        "building_prior_uint8": prior["building_prob_uint8"],
        "pred_building_prob": prior["building_prob"],
        "pred_building_binary": prior["building_binary"],
        "pred_building_prob_uint8": prior["building_prob_uint8"],
        "prior_type": "event_excluded_oof_probability",
        "outer_fold": outer,
        "inner_role": role,
        "prior_model_id": model["prior_model_id"],
        "prior_excluded_events": model["excluded_events"],
        "prior_train_manifest_sha256": model["train_manifest_sha256"],
        "prior_export_manifest_sha256": model["export_manifest_sha256"],
    })
    return output


def tune_threshold(rows: list[dict[str, Any]], data_root: Path) -> dict[str, Any]:
    grid = np.arange(0.30, 0.701, 0.05)
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((len(grid), 3), dtype=np.int64))
    for row in rows:
        with np.load(resolve(data_root, str(row["building_prior"]))) as payload:
            prob = np.asarray(payload["prob"], dtype=np.float32)
        target = np.asarray(Image.open(resolve(data_root, str(row["oracle_building_mask"]))).convert("L")) > 0
        event = str(row["event_group_id"])
        for index, threshold in enumerate(grid):
            pred = prob >= threshold
            counts[event][index] += (np.count_nonzero(pred & target), np.count_nonzero(pred & ~target), np.count_nonzero(~pred & target))
    scores = []
    for index, threshold in enumerate(grid):
        event_f1 = []
        for event in sorted(counts):
            tp, fp, fn = (int(value) for value in counts[event][index])
            event_f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        scores.append({"threshold": float(round(threshold, 2)), "event_macro_building_f1": float(np.mean(event_f1)), "event_f1": dict(zip(sorted(counts), event_f1))})
    selected = max(scores, key=lambda row: (row["event_macro_building_f1"], -abs(row["threshold"] - 0.5)))
    return {"grid": scores, "selected_threshold": selected["threshold"], "selection": "max event-macro building F1; tie closest to 0.5"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--stage1-queue-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol_root = args.protocol_root.resolve()
    queue_root = args.stage1_queue_root.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite finalized manifests: {output_root}")
    lineage = read_json(protocol_root / "prior_lineage.json")
    models = {key_for(set(model["excluded_events"])): model for model in lineage["models"]}
    prior_indexes: dict[str, dict[str, dict[str, Any]]] = {}
    for model in lineage["models"]:
        model_id = str(model["prior_model_id"])
        completed = queue_root / "runs" / model_id / "completed.json"
        index_path = queue_root / "priors" / model_id / "prior_manifest_index.jsonl"
        if not completed.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"incomplete Stage-1 lineage model: {model_id}")
        index = {str(row["id"]): row for row in read_jsonl(index_path)}
        if len(index) != int(model["export_count"]):
            raise ValueError(f"prior export count mismatch for {model_id}")
        prior_indexes[model_id] = index

    protocol = read_json(protocol_root / "protocol.json")
    fold_pairs = [set(fold["outer_events"]) for fold in protocol["folds"]]
    all_events = set().union(*fold_pairs)
    audit_rows: list[dict[str, Any]] = []
    for outer, outer_events in enumerate(fold_pairs):
        inner_events = fold_pairs[(outer + 1) % 7]
        base_root = protocol_root / "folds" / f"outer_{outer}" / "stage2_base"
        base_by_role = {role: read_jsonl(base_root / f"{role}.jsonl") for role in ("train", "inner_val", "outer_eval")}

        role_specs: dict[str, tuple[list[dict[str, Any]], Any]] = {
            "outer_eval": (base_by_role["outer_eval"], lambda event: outer_events),
            "inner_val": (base_by_role["inner_val"], lambda event: outer_events | inner_events),
            "inner_train": (base_by_role["train"], lambda event: outer_events | inner_events | next(pair for pair in fold_pairs if event in pair)),
            "final_train": (base_by_role["train"] + base_by_role["inner_val"], lambda event: outer_events | next(pair for pair in fold_pairs if event in pair)),
        }
        finalized: dict[str, list[dict[str, Any]]] = {}
        for role, (base_rows, exclusion_fn) in role_specs.items():
            role_rows = []
            for row in base_rows:
                event = str(row["event_group_id"])
                excluded = set(exclusion_fn(event))
                model = models[key_for(excluded)]
                prior = prior_indexes[str(model["prior_model_id"])].get(str(row["id"]))
                if prior is None:
                    raise KeyError(f"missing OOF prior for outer={outer} role={role} id={row['id']}")
                role_rows.append(attach(row, prior, model, outer, role))
            finalized[role] = sorted(role_rows, key=lambda row: (str(row["event_group_id"]), str(row["id"])))
            path = output_root / f"outer_{outer}" / f"{role}.jsonl"
            write_jsonl(path, finalized[role])
            audit_rows.append({"outer_fold": outer, "role": role, "count": len(role_rows), "manifest": str(path), "sha256": sha256_file(path)})
        threshold = tune_threshold(finalized["final_train"], data_root)
        write_json(output_root / f"outer_{outer}" / "building_threshold.json", threshold)

    write_json(output_root / "FINALIZED.json", {
        "status": "finalized",
        "evidence_scope": "nested_cv_development_only_no_blind_confirmation",
        "fold_roles": audit_rows,
        "hard_error_count": 0,
    })
    print(json.dumps({"status": "finalized", "fold_role_count": len(audit_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
