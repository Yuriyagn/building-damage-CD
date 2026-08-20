from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_rq2_blind_derangement import build_mapping, read_blind_manifest  # noqa: E402
from run_rq2_model_development import (  # noqa: E402
    candidate_config,
    optional_blind_lock_summary,
    read_selected_steps,
)
from summarize_rq2_model_development import full_gates, pilot_gates  # noqa: E402
from validate_rq2_blind_evaluator_lock import load_and_validate, validate_lock  # noqa: E402
from validate_rq2_blind_prediction_package import validate_package  # noqa: E402


def valid_lock() -> dict:
    return {
        "protocol_id": "rq2_model_development_v1.0",
        "status": "accepted",
        "evaluator": "Independent Custodian",
        "accepted_at": "2026-08-18T12:00:00Z",
        "written_confirmation_sha256": "5" * 64,
        "event_count": 5,
        "event_ids_sha256": "1" * 64,
        "input_manifest_sha256": "2" * 64,
        "hidden_labels_sha256": "3" * 64,
        "metric_script_sha256": "4" * 64,
        "labels_withheld": True,
        "pixel_semantic_evaluation": True,
        "damaged_event_count": 2,
        "destroyed_event_count": 2,
        "geospatial_overlap_audit": "custodian_attested_passed",
        "excluded_event_overlap": False,
        "format_dry_run_limit": 1,
        "scored_submission_limit": 1,
        "resubmission_policy": "only_if_no_score_was_computed_or_disclosed",
    }


class RQ2BlindLockTest(unittest.TestCase):
    def test_valid_lock_passes_and_template_digests_fail(self) -> None:
        self.assertEqual(validate_lock(valid_lock()), [])
        template = valid_lock()
        template["event_ids_sha256"] = "0" * 64
        self.assertTrue(any("zero digest" in error for error in validate_lock(template)))

    def test_missing_lock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                load_and_validate(Path(tmp) / "missing.json")


class RQ2QueueProtocolTest(unittest.TestCase):
    def test_internal_development_does_not_require_blind_lock(self) -> None:
        summary = optional_blind_lock_summary(None)
        self.assertEqual(summary["status"], "not_provided")
        self.assertFalse(summary["required_for_internal_development"])
        self.assertFalse(summary["blind_confirmation_completed"])

    def test_frozen_steps_reject_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = (14000, 10000, 14000, 14000, 14000, 12000, 16000)
            for outer, step in enumerate(steps):
                (root / f"outer_{outer}.json").write_text(
                    json.dumps(
                        {
                            "selected_step": step,
                            "outer_metrics_used": False,
                        }
                    )
                )
            self.assertEqual(read_selected_steps(root)[6], 16000)
            (root / "outer_6.json").write_text(
                json.dumps({"selected_step": 18000, "outer_metrics_used": False})
            )
            with self.assertRaisesRegex(ValueError, "anchor drifted"):
                read_selected_steps(root)

    def test_candidate_configs_change_one_factor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = root / "manifests" / "outer_0"
            derangements = root / "derangements" / "outer_0"
            manifests.mkdir(parents=True)
            derangements.mkdir(parents=True)
            (manifests / "building_threshold.json").write_text('{"selected_threshold": 0.7}')
            for candidate in ("R2_A", "R2_B", "R2_C"):
                cfg = candidate_config(
                    manifest_root=root / "manifests",
                    derangement_root=root / "derangements",
                    outer=0,
                    candidate=candidate,
                    condition="C3",
                    seed=42,
                    selected_step=14000,
                )
                self.assertEqual(cfg["dataset"]["input_mode"], "pre_prior_sar5")
                self.assertEqual(cfg["train"]["max_optimizer_steps"], 14000)
                self.assertIn("train_sar_permutation_file", cfg["dataset"])
            self.assertEqual(
                candidate_config(
                    manifest_root=root / "manifests",
                    derangement_root=root / "derangements",
                    outer=0,
                    candidate="R2_C",
                    condition="C2",
                    seed=42,
                    selected_step=14000,
                )["train"]["event_sampling_max_weight"],
                4.0,
            )


class RQ2BlindPackageTest(unittest.TestCase):
    def write_manifest(self, root: Path, include_label: bool = False) -> Path:
        rows = []
        for event in ("new_a", "new_b", "new_c"):
            for index in range(2):
                row = {
                    "id": f"{event}_{index}",
                    "event_id": event,
                    "pre_image": f"pre/{event}_{index}.png",
                    "post_sar": f"sar/{event}_{index}.png",
                    "width": 4,
                    "height": 3,
                    "sar_sensor": "sensor",
                }
                if include_label:
                    row["mask"] = "forbidden.png"
                rows.append(row)
        manifest = root / "blind.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return manifest

    def test_blind_manifest_rejects_labels_and_mapping_is_bijective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.write_manifest(root)
            rows = read_blind_manifest(manifest)
            first, _ = build_mapping(rows, seed=3407)
            second, _ = build_mapping(list(reversed(rows)), seed=3407)
            self.assertEqual(set(first), set(first.values()))
            self.assertTrue(all(target != source for target, source in first.items()))
            self.assertEqual(set(second), set(second.values()))
            with self.assertRaisesRegex(ValueError, "exposes label fields"):
                read_blind_manifest(self.write_manifest(root, include_label=True))

    def test_synthetic_prediction_tree_passes_format_only_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.write_manifest(root)
            rows = read_blind_manifest(manifest)
            for method in ("F0", "candidate"):
                for condition in ("C2", "C3"):
                    for seed in (42, 3407, 2026):
                        directory = root / "package" / "predictions" / method / condition / f"seed_{seed}"
                        directory.mkdir(parents=True)
                        for row in rows:
                            Image.fromarray(np.full((3, 4), seed % 4, dtype=np.uint8)).save(
                                directory / f"{row['id']}.png"
                            )
            audit = validate_package(manifest, root / "package")
            self.assertEqual(audit["status"], "format_valid_no_scores_computed")
            self.assertEqual(audit["prediction_count"], len(rows) * 12)
            invalid = (
                root
                / "package"
                / "predictions"
                / "F0"
                / "C2"
                / "seed_42"
                / f"{rows[0]['id']}.png"
            )
            Image.fromarray(np.ones((3, 4), dtype=np.uint16)).save(invalid)
            with self.assertRaisesRegex(ValueError, "expected uint8 PNG"):
                validate_package(manifest, root / "package")


class RQ2GateTest(unittest.TestCase):
    def summary(self) -> dict:
        return {
            "candidate_minus_f0_event_macro_damage": 0.03,
            "candidate_minus_f0_positive_event_count": 11,
            "candidate_minus_f0_positive_seed_count": 3,
            "candidate_minus_f0_damaged": 0.01,
            "candidate_minus_f0_destroyed": 0.02,
            "candidate_c2_minus_c3_event_macro_damage": 0.04,
            "candidate_c2_minus_c3_positive_seed_count": 3,
            "worst_event_candidate_minus_f0_grade": -0.005,
            "worst_event_candidate_c2_minus_c3_grade": -0.005,
        }

    def test_frozen_pilot_and_full_gates(self) -> None:
        self.assertTrue(all(pilot_gates(self.summary()).values()))
        self.assertTrue(all(full_gates(self.summary()).values()))
        failed = self.summary()
        failed["worst_event_candidate_c2_minus_c3_grade"] = -0.02
        self.assertFalse(all(full_gates(failed).values()))


if __name__ == "__main__":
    unittest.main()
