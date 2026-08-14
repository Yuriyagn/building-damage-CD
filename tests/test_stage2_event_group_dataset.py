from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_stage2_event_group_dataset import (  # noqa: E402
    BrightSample,
    load_mapping,
    load_split_assignments,
    rasterize_annotations,
    rectangles_overlap,
    validate_event_groups,
)


class EventGroupConfigurationTest(unittest.TestCase):
    def test_mapping_and_split_cover_fourteen_real_event_groups(self) -> None:
        mapping = load_mapping(REPO_ROOT / "configs/stage2_v2/event_group_mapping_v1.csv")
        plan = json.loads(
            (REPO_ROOT / "configs/stage2_v2/event_group_split_v1.json").read_text(encoding="utf-8")
        )
        assignments = load_split_assignments(plan)
        current_groups = {
            row["event_group_id"]
            for (dataset, _), row in mapping.items()
            if dataset == "disasterm3_minimal_v0.2"
        }
        self.assertEqual(len(current_groups), 14)
        self.assertEqual(set(assignments), current_groups)
        self.assertEqual(assignments["turkey_eq_2023"], "train")
        self.assertEqual(mapping[("bright_cvprw26", "congo-volcano")]["event_group_id"], "nyiragongo_2021")

    def test_event_group_overlap_is_rejected(self) -> None:
        rows = {
            "train": [{"event_group_id": "same"}],
            "val": [{"event_group_id": "same"}],
            "test": [],
        }
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_event_groups(rows)


class BrightConversionTest(unittest.TestCase):
    def test_polygon_categories_rasterize_to_semantic_values(self) -> None:
        sample = BrightSample(
            source_split="train",
            image_id=1,
            sample_id="event_00000001",
            source_event_name="event",
            pre_image=Path("pre.tif"),
            post_sar=Path("post.tif"),
            target_json=Path("target.json"),
            width=8,
            height=8,
            annotations=(
                {"category_id": 1, "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]]},
                {"category_id": 2, "segmentation": [[4, 1, 6, 1, 6, 3, 4, 3]]},
                {"category_id": 3, "segmentation": [[2, 4, 5, 4, 5, 6, 2, 6]]},
            ),
            geographic_footprint={"epsg": 1, "bbox_projected": [0, 0, 8, 8]},
        )
        mask = rasterize_annotations(sample)
        self.assertEqual(mask.shape, (8, 8))
        self.assertEqual(set(int(value) for value in np.unique(mask)), {0, 1, 2, 3})

    def test_projected_footprint_overlap_requires_same_epsg_and_positive_area(self) -> None:
        left = {"epsg": 32637, "bbox_projected": [0, 0, 10, 10]}
        overlap = {"epsg": 32637, "bbox_projected": [5, 5, 12, 12]}
        touching = {"epsg": 32637, "bbox_projected": [10, 0, 20, 10]}
        other_crs = {"epsg": 32636, "bbox_projected": [5, 5, 12, 12]}
        self.assertEqual(rectangles_overlap(left, overlap), (True, 25.0))
        self.assertEqual(rectangles_overlap(left, touching), (False, 0.0))
        self.assertEqual(rectangles_overlap(left, other_crs), (False, 0.0))


if __name__ == "__main__":
    unittest.main()
