from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from overlap_audit_core import (  # noqa: E402
    FeatureCache,
    LocalFeatureCache,
    Sample,
    feature_from_array,
    hamming,
    local_descriptor_candidate_pairs,
    local_geometry,
    patch_overlap,
)


class PartialOverlapAuditTest(unittest.TestCase):
    def test_hash_is_stable_and_brightness_tolerant(self) -> None:
        rng = np.random.default_rng(7)
        image = rng.integers(0, 230, size=(256, 256), dtype=np.uint8)
        brighter = np.clip(image.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        base = feature_from_array(image)
        changed = feature_from_array(brighter)
        self.assertLessEqual(hamming(base.phash, changed.phash), 8)

    def test_patch_overlap_detects_shared_quadrant(self) -> None:
        rng = np.random.default_rng(11)
        shared = rng.integers(0, 255, size=(128, 128), dtype=np.uint8)
        left = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
        right = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
        left[:128, :128] = shared
        right[128:, 128:] = shared
        score, count, best = patch_overlap(
            feature_from_array(left).patches, feature_from_array(right).patches
        )
        self.assertGreater(best, 0.99)
        self.assertGreater(score, 0.3)
        self.assertGreaterEqual(count, 1)

    def test_feature_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "image.png"
            image = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
            cv2.imwrite(str(path), image)
            cache = FeatureCache(root / "cache")
            first = cache.get(path)
            second = cache.get(path)
            self.assertEqual(first.phash, second.phash)
            np.testing.assert_allclose(first.embedding, second.embedding)
            np.testing.assert_allclose(first.patches, second.patches)

    def test_orb_candidate_retrieval_finds_shifted_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(19)
            image = rng.integers(0, 255, size=(512, 512), dtype=np.uint8)
            shifted = cv2.warpAffine(
                image,
                np.float32([[1, 0, 80], [0, 1, 40]]),
                (512, 512),
                borderMode=cv2.BORDER_REFLECT_101,
            )
            unrelated = rng.integers(0, 255, size=(512, 512), dtype=np.uint8)
            paths = []
            for name, value in (("base", image), ("shifted", shifted), ("unrelated", unrelated)):
                path = root / f"{name}.png"
                cv2.imwrite(str(path), value)
                paths.append(path)

            def sample(name: str, path: Path, split: str) -> Sample:
                return Sample(split, name, "event", "type", "region", path, path, path)

            pairs = local_descriptor_candidate_pairs(
                [sample("base", paths[0], "train")],
                [sample("shifted", paths[1], "test"), sample("unrelated", paths[2], "test")],
                LocalFeatureCache(max_side=512),
            )
            self.assertIn((0, 0), pairs)
            self.assertNotIn((0, 1), pairs)

    def test_geometry_accepts_translation_and_rejects_unrelated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(31)
            image = rng.integers(0, 255, size=(512, 512), dtype=np.uint8)
            shifted = cv2.warpAffine(
                image, np.float32([[1, 0, 64], [0, 1, 32]]), (512, 512), borderMode=cv2.BORDER_REFLECT_101
            )
            unrelated = rng.integers(0, 255, size=(512, 512), dtype=np.uint8)
            paths = []
            for name, value in (("base", image), ("shift", shifted), ("other", unrelated)):
                path = root / f"{name}.png"
                cv2.imwrite(str(path), value)
                paths.append(path)
            cache = LocalFeatureCache(max_side=512)
            _, inliers, _, transform, score = local_geometry(paths[0], paths[1], cache)
            self.assertGreater(inliers, 12)
            self.assertGreater(score, 0.7)
            self.assertNotEqual(transform, "none_or_degenerate")
            _, other_inliers, _, _, other_score = local_geometry(paths[0], paths[2], cache)
            self.assertLess(other_inliers, 12)
            self.assertLess(other_score, 0.5)


if __name__ == "__main__":
    unittest.main()
