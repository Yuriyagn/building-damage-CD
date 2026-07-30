from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
REPO_ROOT = SRC.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.datasets import Stage2DamageDataset  # noqa: E402
from stage2.datasets_v2 import build_derangement  # noqa: E402
from stage2.losses_v2 import BuildingOnlyGradeBinaryAuxLoss, BuildingOnlyGradeLoss  # noqa: E402
from stage2.metrics_v2 import Stage2V2MeterBundle  # noqa: E402
from stage2.summarize_stage2_v2 import infer_run, paired_bootstrap  # noqa: E402
from stage2.test_stage2_v2 import decode_grade_logits  # noqa: E402
from stage2.train_stage2_v2 import audit_state, checkpoint_metrics_for_policy  # noqa: E402
from models.external_uabcd import UABCDInputAdapter  # noqa: E402
from scripts.prune_workspace_artifacts import checkpoint_is_retained  # noqa: E402


class BuildingOnlyLossTest(unittest.TestCase):
    def test_background_has_zero_gradient(self) -> None:
        logits = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]], [[-1.0, -1.0], [-1.0, -1.0]]]],
            requires_grad=True,
        )
        target = torch.tensor([[[0, 1], [2, 3]]])
        loss = BuildingOnlyGradeLoss()(logits, target)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[:, :, 0, 0] == 0))
        self.assertGreater(float(logits.grad[:, :, 0, 1].abs().sum()), 0.0)

    def test_all_background_is_differentiable_zero(self) -> None:
        logits = torch.randn(2, 3, 4, 4, requires_grad=True)
        loss = BuildingOnlyGradeLoss()(logits, torch.zeros(2, 4, 4, dtype=torch.long))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_binary_auxiliary_loss_is_masked_to_buildings(self) -> None:
        logits = torch.randn(1, 3, 3, 3, requires_grad=True)
        target = torch.tensor([[[0, 1, 2], [3, 0, 1], [2, 3, 0]]])
        loss = BuildingOnlyGradeBinaryAuxLoss(BuildingOnlyGradeLoss(), binary_aux_weight=0.3)(logits, target)
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        background = (target == 0).unsqueeze(1).expand_as(logits)
        self.assertTrue(torch.all(logits.grad[background] == 0))


class FixedDerangementTest(unittest.TestCase):
    def test_within_event_is_fixed_and_has_no_self_pairs(self) -> None:
        rows = [
            {"id": f"a{index}", "event_id": "a"} for index in range(4)
        ] + [{"id": f"b{index}", "event_id": "b"} for index in range(3)]
        first, info = build_derangement(rows, "within_event", seed=3407)
        second, _ = build_derangement(rows, "within_event", seed=3407)
        self.assertEqual(first, second)
        self.assertTrue(all(target != source for target, source in enumerate(first)))
        self.assertTrue(all(rows[target]["event_id"] == rows[source]["event_id"] for target, source in enumerate(first)))
        self.assertEqual(info["mode"], "within_event")

    def test_singleton_event_is_rejected(self) -> None:
        rows = [{"id": "a0", "event_id": "a"}, {"id": "b0", "event_id": "b"}, {"id": "b1", "event_id": "b"}]
        with self.assertRaisesRegex(ValueError, "singleton"):
            build_derangement(rows, "within_event", seed=42)


class ExternalUABCDAdapterTest(unittest.TestCase):
    def test_splits_and_normalizes_the_unified_six_channel_input(self) -> None:
        class RecordingCore(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stream_a: torch.Tensor | None = None
                self.stream_b: torch.Tensor | None = None

            def forward(self, stream_a: torch.Tensor, stream_b: torch.Tensor) -> torch.Tensor:
                self.stream_a = stream_a.detach().clone()
                self.stream_b = stream_b.detach().clone()
                return torch.cat([stream_a[:, :1, ::2, ::2], stream_b[:, :2, ::2, ::2]], dim=1)

        core = RecordingCore()
        model = UABCDInputAdapter(core)
        image = torch.zeros(1, 6, 8, 8)
        image[:, 0:3] = 0.25
        image[:, 3:6] = 0.75
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (1, 3, 8, 8))
        self.assertTrue(torch.allclose(core.stream_a, torch.full((1, 3, 8, 8), -0.5)))
        self.assertTrue(torch.allclose(core.stream_b, torch.full((1, 3, 8, 8), 0.5)))

    def test_primary_checkpoint_policy_retains_one_weight(self) -> None:
        metrics = checkpoint_metrics_for_policy({"checkpoint_policy": "primary_only"})
        self.assertEqual(metrics, {"best_bo_grade_macro_f1.pth": "val_building_only_macro_f1_3class"})

    def test_run_info_uses_explicit_data_audit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            audit_path = Path(tmp_name) / "audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "status": "warning",
                        "hard_error_count": 0,
                        "warning_count": 3,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"STAGE2_DATA_AUDIT_PATH": str(audit_path)},
            ):
                state = audit_state()
        self.assertTrue(state["exists"])
        self.assertEqual(state["path"], str(audit_path.resolve()))
        self.assertEqual(state["hard_error_count"], 0)
        self.assertEqual(state["warning_count"], 3)

    def test_cleanup_retains_only_formal_unified_primary_checkpoint(self) -> None:
        formal = Path(
            "stage2/unified_v1/S2U1_UABCD_paired/seed_42/"
            "run_20260731/checkpoints/best_bo_grade_macro_f1.pth"
        )
        overfit = Path(
            "stage2/unified_v1_overfit/paired_seed42/"
            "checkpoints/best_bo_grade_macro_f1.pth"
        )
        self.assertTrue(checkpoint_is_retained(formal))
        self.assertFalse(checkpoint_is_retained(overfit))


class OGSRFeatureDatasetTest(unittest.TestCase):
    def test_ogsr_texture_mode_loads_and_crops_feature_stack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            (root / "images" / "train").mkdir(parents=True)
            (root / "masks_4class" / "train").mkdir(parents=True)
            (root / "building_priors" / "train").mkdir(parents=True)
            feature_root = root / "ogsr_features"
            (feature_root / "features" / "train").mkdir(parents=True)

            pre = np.zeros((8, 8, 3), dtype=np.uint8)
            pre[..., 0] = 80
            sar = np.arange(64, dtype=np.uint8).reshape(8, 8)
            mask = np.ones((8, 8), dtype=np.uint8)
            mask[0:2, 0:2] = 0
            mask[4:6, 4:6] = 2
            prior = np.ones((8, 8), dtype=np.float32)
            Image.fromarray(pre).save(root / "images" / "train" / "sample_pre.png")
            Image.fromarray(sar).save(root / "images" / "train" / "sample_sar_post.png")
            Image.fromarray(mask).save(root / "masks_4class" / "train" / "sample.png")
            np.savez_compressed(root / "building_priors" / "train" / "sample.npz", prob=prior)

            feature_path = feature_root / "features" / "train" / "sample.npz"
            feature_arrays = {
                "expected_sar": np.full((8, 8), 0.2, dtype=np.float32),
                "expected_sar_grad": np.full((8, 8), 0.1, dtype=np.float32),
                "residual_sar": np.full((8, 8), 0.3, dtype=np.float32),
                "abs_residual_sar": np.full((8, 8), 0.3, dtype=np.float32),
                "residual_grad": np.full((8, 8), -0.1, dtype=np.float32),
                "abs_residual_grad": np.full((8, 8), 0.1, dtype=np.float32),
            }
            np.savez_compressed(feature_path, **feature_arrays)
            (feature_root / "index_train.json").write_text(
                json.dumps({"features": {"sample": "features/train/sample.npz"}}),
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "split": "train",
                        "event_id": "event",
                        "pre_image": "images/train/sample_pre.png",
                        "post_sar": "images/train/sample_sar_post.png",
                        "mask_multiclass": "masks_4class/train/sample.png",
                        "building_prior": "building_priors/train/sample.npz",
                        "pred_building_prob": "building_priors/train/sample.npz",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = Stage2DamageDataset(
                data_root=root,
                manifest=manifest,
                train=True,
                prior_type="predicted",
                input_mode="pre_sar_ogsr_texture_prior",
                crop_size=4,
                crop_probabilities={"random": 1.0},
                hflip=False,
                vflip=False,
                rotate90=False,
                ogsr_feature_root=feature_root,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["image"].shape), (12, 4, 4))
            self.assertEqual(tuple(item["ogsr_features"].shape), (6, 4, 4))
            self.assertEqual(tuple(item["sar_grad"].shape), (1, 4, 4))

            additive_dataset = Stage2DamageDataset(
                data_root=root,
                manifest=manifest,
                train=True,
                prior_type="predicted",
                input_mode="pre_sar_texture_ogsr_add_prior",
                crop_size=4,
                crop_probabilities={"random": 1.0},
                hflip=False,
                vflip=False,
                rotate90=False,
                ogsr_feature_root=feature_root,
            )
            additive_item = additive_dataset[0]
            self.assertEqual(tuple(additive_item["image"].shape), (14, 4, 4))


class V2MetricTest(unittest.TestCase):
    def test_summary_infers_phase2_experiment(self) -> None:
        experiment, seed = infer_run(
            Path("outputs/stage2/v2_phase2/V2_B2_focal_sar_predicted_prior/seed_42/run_x/val_best_grade/metrics.json")
        )
        self.assertEqual(experiment, "V2_B2_focal_sar_predicted_prior")
        self.assertEqual(seed, 42)

    def test_grade_and_predicted_gate_are_separate(self) -> None:
        target = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        grade = np.array([[2, 1], [2, 3]], dtype=np.uint8)
        support = np.array([[False, True], [False, True]])
        meter = Stage2V2MeterBundle()
        meter.update(grade, target, support)
        metrics = meter.compute()
        self.assertEqual(metrics["building_only_macro_f1_3class"], 1.0)
        self.assertEqual(metrics["oracle_gate_macro_f1_4class"], 1.0)
        self.assertLess(metrics["predicted_gate_macro_f1_4class"], 1.0)
        self.assertAlmostEqual(metrics["predicted_gate_f1_background"], 2.0 / 3.0)

    def test_damage_margin_threshold_decoding(self) -> None:
        logits = torch.tensor(
            [
                [
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[0.9, 1.3, 0.1, 0.2]],
                    [[0.2, 0.1, 0.8, 1.4]],
                ]
            ]
        )
        self.assertEqual(decode_grade_logits(logits).tolist(), [[[1, 2, 1, 3]]])
        self.assertEqual(decode_grade_logits(logits, damage_margin_threshold=0.0).tolist(), [[[1, 2, 1, 3]]])
        self.assertEqual(decode_grade_logits(logits, damage_margin_threshold=-0.25).tolist(), [[[2, 2, 3, 3]]])

    def test_paired_bootstrap_intersects_sample_keys(self) -> None:
        perfect = "[10,0,0,0,10,0,0,0,10]"
        all_intact = "[10,0,0,10,0,0,10,0,0]"
        left = {
            "seed": 42,
            "samples": [
                {"split": "test", "id": "shared", "bo_confusion_3x3": perfect},
                {"split": "test", "id": "left_only", "bo_confusion_3x3": perfect},
            ],
        }
        right = {
            "seed": 42,
            "samples": [
                {"split": "test", "id": "shared", "bo_confusion_3x3": all_intact},
            ],
        }
        rows = paired_bootstrap([left], [right], iterations=20, seed=1)
        self.assertEqual(rows[0]["common_sample_count_min"], 1)
        self.assertGreater(rows[0]["point_difference"], 0.0)
        self.assertIn("bo_intact_f1", {row["metric"] for row in rows})


if __name__ == "__main__":
    unittest.main()
