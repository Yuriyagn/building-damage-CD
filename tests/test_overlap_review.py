from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from apps.overlap_review.db import CandidateStore


class OverlapReviewAppTest(unittest.TestCase):
    def test_summary_query_and_decision_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fields = [
                "pair_id", "split_pair", "risk_level", "overlap_score",
                "split_a", "id_a", "event_a", "disaster_a", "region_a", "pre_a", "sar_a", "mask_a",
                "split_b", "id_b", "event_b", "disaster_b", "region_b", "pre_b", "sar_b", "mask_b",
                "review_label", "review_action", "review_comment",
            ]
            row = {field: "" for field in fields}
            row.update(
                {
                    "pair_id": "pair1", "split_pair": "train-test", "risk_level": "high",
                    "overlap_score": "0.95", "split_a": "train", "id_a": "a", "event_a": "ea",
                    "split_b": "test", "id_b": "b", "event_b": "eb",
                }
            )
            with (root / "candidates_all.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            (root / "review_decisions.csv").write_text(
                "pair_id,review_label,review_action,review_comment,reviewed_at\n", encoding="utf-8"
            )
            store = CandidateStore(root)
            self.assertEqual(store.summary()["total"], 1)
            items = store.query("train-test", "high", "unreviewed", 0, 50)["items"]
            self.assertEqual(len(items), 1)
            response = store.save_decision(
                "pair1",
                "confirmed_overlap",
                "exclude_train_side",
                "synthetic test",
            )
            self.assertEqual(response["review_status"], "reviewed")
            self.assertEqual(store.summary()["review_status"]["reviewed"], 1)
            self.assertEqual(store.summary()["mandatory_train_test"]["unreviewed"], 0)

    def test_not_overlap_keeps_both_sides(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = root / "manifests"
            manifests.mkdir()
            for split in ("train", "val", "test"):
                (manifests / f"stage2_master_{split}.jsonl").write_text(
                    json.dumps({"id": f"{split}_a", "split": split}) + "\n", encoding="utf-8"
                )
            candidates = root / "candidates.csv"
            with candidates.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["pair_id", "split_pair", "risk_level", "split_a", "id_a", "split_b", "id_b"])
                writer.writeheader()
                writer.writerow({"pair_id":"pair1", "split_pair":"train-test", "risk_level":"medium", "split_a":"train", "id_a":"train_a", "split_b":"test", "id_b":"test_a"})
            decisions = root / "decisions.csv"
            with decisions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["pair_id", "review_label", "review_action", "review_comment", "reviewed_at"])
                writer.writeheader()
                writer.writerow({"pair_id":"pair1", "review_label":"not_overlap", "review_action":"keep_both_record_only", "review_comment":"no-data false positive", "reviewed_at":"now"})
            out = root / "reviewed"
            script = Path(__file__).resolve().parents[1] / "scripts/apply_overlap_decisions.py"
            subprocess.run([sys.executable, str(script), "--manifest-root", str(manifests), "--decisions", str(decisions), "--candidates", str(candidates), "--policy", "relaxed_train_test_only", "--out-root", str(out)], check=True)
            self.assertIn("train_a", (out / "stage2_master_train.jsonl").read_text())
            self.assertIn("test_a", (out / "stage2_master_test.jsonl").read_text())
            summary = json.loads((out / "overlap_review_summary.json").read_text())
            self.assertEqual(summary["removed_unique_sample_count"], 0)

    def test_apply_decisions_excludes_train_side(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = root / "manifests"
            manifests.mkdir()
            rows = {
                "train": {"id": "train_a", "split": "train"},
                "val": {"id": "val_a", "split": "val"},
                "test": {"id": "test_a", "split": "test"},
            }
            for split, row in rows.items():
                (manifests / f"stage2_master_{split}.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )
            candidates = root / "candidates_all.csv"
            candidate_fields = [
                "pair_id", "split_pair", "risk_level", "split_a", "id_a", "split_b", "id_b"
            ]
            with candidates.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=candidate_fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "pair1", "split_pair": "train-test", "risk_level": "high",
                        "split_a": "train", "id_a": "train_a", "split_b": "test", "id_b": "test_a",
                    }
                )
            decisions = root / "review_decisions.csv"
            decision_fields = ["pair_id", "review_label", "review_action", "review_comment", "reviewed_at"]
            with decisions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=decision_fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "pair1", "review_label": "confirmed_overlap",
                        "review_action": "exclude_train_side", "review_comment": "test", "reviewed_at": "now",
                    }
                )
            out = root / "reviewed"
            script = Path(__file__).resolve().parents[1] / "scripts/apply_overlap_decisions.py"
            subprocess.run(
                [
                    sys.executable, str(script), "--manifest-root", str(manifests),
                    "--decisions", str(decisions), "--candidates", str(candidates),
                    "--policy", "relaxed_train_test_only", "--out-root", str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual((out / "stage2_master_train.jsonl").read_text(), "")
            self.assertIn("test_a", (out / "stage2_master_test.jsonl").read_text())
            summary = json.loads((out / "overlap_review_summary.json").read_text())
            self.assertEqual(summary["removed_unique_sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
