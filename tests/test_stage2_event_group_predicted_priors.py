from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finalize_stage2_event_group_predicted_priors import (  # noqa: E402
    predicted_row,
    require_exact_id_coverage,
    validate_event_groups,
)


class PredictedPriorManifestTest(unittest.TestCase):
    def test_predicted_row_uses_float_probability_as_model_input(self) -> None:
        row = {"id": "sample", "event_group_id": "event", "stage1_model": None}
        prior = {
            "building_prob": "/tmp/sample.npz",
            "building_prob_uint8": "/tmp/sample_prob.png",
            "building_binary": "/tmp/sample_binary.png",
        }
        output = predicted_row(row, prior, 0.6)
        self.assertEqual(output["building_prior"], prior["building_prob"])
        self.assertEqual(output["building_prior_binary"], prior["building_binary"])
        self.assertEqual(output["prior_type"], "predicted")
        self.assertEqual(output["stage1_threshold"], 0.6)

    def test_exact_id_coverage_rejects_missing_or_extra_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            require_exact_id_coverage({"a", "b"}, {"a", "c"}, "test")

    def test_event_group_overlap_is_rejected(self) -> None:
        rows = {
            "train": [{"event_group_id": "same"}],
            "val": [{"event_group_id": "same"}],
            "test": [],
        }
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_event_groups(rows)


if __name__ == "__main__":
    unittest.main()
