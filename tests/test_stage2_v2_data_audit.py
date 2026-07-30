from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_stage2_v2_data import build_report  # noqa: E402


class Stage2V2DataAuditTest(unittest.TestCase):
    def test_reports_distribution_and_event_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = root / "manifests"
            manifests.mkdir()
            masks = {
                "train": np.array([[0, 1], [2, 3]], dtype=np.uint8),
                "val": np.array([[0, 1], [2, 2]], dtype=np.uint8),
                "test": np.array([[0, 1], [3, 3]], dtype=np.uint8),
            }
            for split, mask in masks.items():
                mask_path = root / f"{split}_mask.png"
                Image.fromarray(mask).save(mask_path)
                row = {
                    "id": f"{split}_sample",
                    "pre_image": f"{split}_pre.png",
                    "post_sar": f"{split}_sar.png",
                    "mask_multiclass": mask_path.name,
                    "event_id": "shared_event" if split != "test" else "test_event",
                    "disaster_type": "earthquake",
                    "country_or_region": "region",
                    "qc_label": "ok",
                }
                (manifests / f"stage2_master_{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = build_report(root, manifests)

            self.assertEqual(report["status"], "warning")
            self.assertEqual(report["hard_error_count"], 0)
            self.assertEqual(report["splits"]["train"]["class_pixels"]["destroyed"], 1)
            self.assertEqual(report["overlaps"]["train_vs_val"]["metadata"]["event_id"], ["shared_event"])

    def test_fails_on_cross_split_id_and_path_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = root / "manifests"
            manifests.mkdir()
            Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(root / "shared_mask.png")
            for split in ("train", "val", "test"):
                row = {
                    "id": "shared" if split != "test" else "test",
                    "pre_image": f"{split}_pre.png",
                    "post_sar": "shared_sar.png" if split != "test" else "test_sar.png",
                    "mask_multiclass": "shared_mask.png" if split != "test" else "test_mask.png",
                    "event_id": split,
                    "disaster_type": "earthquake",
                    "country_or_region": "region",
                    "qc_label": "ok",
                }
                if split == "test":
                    Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(root / "test_mask.png")
                (manifests / f"stage2_master_{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = build_report(root, manifests)

            self.assertEqual(report["status"], "fail")
            self.assertGreaterEqual(report["hard_error_count"], 2)

    def test_fails_on_renamed_cross_split_content_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = root / "manifests"
            manifests.mkdir()
            shared = {
                "pre_image": np.array([[4, 5]], dtype=np.uint8),
                "post_sar": np.array([[6, 7]], dtype=np.uint8),
                "mask_multiclass": np.array([[1, 3]], dtype=np.uint8),
            }
            for split in ("train", "val", "test"):
                row = {
                    "id": f"{split}_renamed_id",
                    "event_id": f"{split}_event",
                    "disaster_type": "earthquake",
                    "country_or_region": "region",
                    "qc_label": "ok",
                }
                for field, array in shared.items():
                    path = root / f"{split}_{field}.png"
                    Image.fromarray(array if split != "val" else array + 1).save(path)
                    row[field] = path.name
                (manifests / f"stage2_master_{split}.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )

            report = build_report(root, manifests)

            self.assertEqual(report["status"], "fail")
            full = report["content_hash_audit"]["full_sample"]
            self.assertEqual(full["duplicate_group_count"], 1)
            self.assertEqual(full["cross_split_group_count"], 1)
            self.assertIn("global full-sample content duplicates", "\n".join(report["hard_errors"]))

    def test_require_event_disjoint_promotes_overlap_to_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = root / "manifests"
            manifests.mkdir()
            for index, split in enumerate(("train", "val", "test")):
                row = {
                    "id": f"{split}_sample",
                    "event_id": "shared_event" if split != "val" else "val_event",
                    "disaster_type": "earthquake",
                    "country_or_region": "region",
                    "qc_label": "ok",
                }
                for field in ("pre_image", "post_sar", "mask_multiclass"):
                    path = root / f"{split}_{field}.png"
                    Image.fromarray(np.array([[index, index + 1]], dtype=np.uint8)).save(path)
                    row[field] = path.name
                (manifests / f"stage2_master_{split}.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )

            report = build_report(root, manifests, require_event_disjoint=True)

            self.assertEqual(report["status"], "fail")
            self.assertIn("event_id values overlap", "\n".join(report["hard_errors"]))


if __name__ == "__main__":
    unittest.main()
