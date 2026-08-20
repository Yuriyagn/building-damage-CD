from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.build_model import build_model  # noqa: E402
from models.metadata_multitask import unpack_model_output  # noqa: E402
from models.rq2_model_development import (  # noqa: E402
    DualStreamUNet,
    hierarchical_grade_log_probs,
    trainable_parameter_count,
)
from stage2.losses_v2 import BuildingOnlyHierarchicalNLLLoss  # noqa: E402
from stage2.train_stage2_v2 import build_capped_event_sampler  # noqa: E402


class RQ2DualStreamModelTest(unittest.TestCase):
    def test_forward_contract_and_parameter_budget(self) -> None:
        candidate = DualStreamUNet(encoder_weights=None)
        candidate.eval()
        with torch.no_grad():
            output = candidate(torch.rand(1, 5, 64, 64))
        logits, auxiliary = unpack_model_output(output)
        self.assertEqual(tuple(logits.shape), (1, 3, 64, 64))
        self.assertIsNone(auxiliary)

        baseline = build_model(
            {
                "model": {
                    "name": "unet",
                    "encoder": "resnet34",
                    "encoder_weights": None,
                    "in_channels": 5,
                    "out_channels": 3,
                }
            },
            no_pretrained=True,
        )
        ratio = trainable_parameter_count(candidate) / trainable_parameter_count(baseline)
        self.assertGreaterEqual(ratio, 0.85)
        self.assertLessEqual(ratio, 1.15)

    def test_sar_stem_is_rgb_mean_and_parameters_are_independent(self) -> None:
        candidate = DualStreamUNet(encoder_weights=None)
        expected = candidate.optical_encoder.conv1.weight.detach().mean(dim=1, keepdim=True)
        observed = candidate.sar_encoder.conv1.weight.detach()
        self.assertTrue(torch.equal(observed, expected))
        self.assertNotEqual(
            candidate.optical_encoder.layer1[0].conv1.weight.data_ptr(),
            candidate.sar_encoder.layer1[0].conv1.weight.data_ptr(),
        )

    def test_rejects_non_frozen_channel_layout(self) -> None:
        candidate = DualStreamUNet(encoder_weights=None)
        with self.assertRaisesRegex(ValueError, r"expected \[B,5,H,W\]"):
            candidate(torch.rand(1, 6, 64, 64))


class RQ2HierarchicalModelTest(unittest.TestCase):
    def test_factorized_probabilities_are_normalized_and_finite(self) -> None:
        affected = torch.tensor([[[[-100.0, 0.0, 100.0]]]])
        destroyed = torch.tensor([[[[100.0, 0.0, -100.0]]]])
        log_probs = hierarchical_grade_log_probs(affected, destroyed)
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(
            torch.allclose(
                torch.exp(log_probs).sum(dim=1),
                torch.ones_like(affected[:, 0]),
                atol=1e-6,
            )
        )

    def test_hierarchical_nll_masks_background(self) -> None:
        affected = torch.randn(1, 1, 3, 3, requires_grad=True)
        destroyed = torch.randn(1, 1, 3, 3, requires_grad=True)
        log_probs = hierarchical_grade_log_probs(affected, destroyed)
        log_probs.retain_grad()
        target = torch.tensor([[[0, 1, 2], [3, 0, 1], [2, 3, 0]]])
        loss = BuildingOnlyHierarchicalNLLLoss()(log_probs, target)
        loss.backward()
        background = (target == 0).unsqueeze(1).expand_as(log_probs)
        self.assertTrue(torch.all(log_probs.grad[background] == 0))

    def test_model_factory_preserves_three_logit_contract(self) -> None:
        model = build_model(
            {
                "model": {
                    "name": "hierarchical_unet_r34",
                    "encoder_weights": None,
                    "in_channels": 5,
                    "out_channels": 3,
                }
            },
            no_pretrained=True,
        )
        model.eval()
        with torch.no_grad():
            logits, auxiliary = unpack_model_output(model(torch.rand(1, 5, 64, 64)))
        self.assertEqual(tuple(logits.shape), (1, 3, 64, 64))
        self.assertIsNone(auxiliary)
        self.assertTrue(
            torch.allclose(
                torch.exp(logits).sum(dim=1),
                torch.ones_like(logits[:, 0]),
                atol=1e-5,
            )
        )


class RQ2CappedEventSamplerTest(unittest.TestCase):
    class DummyDataset:
        def __init__(self) -> None:
            self.rows = (
                [{"event_id": "tiny"}] * 2
                + [{"event_id": "medium"}] * 8
                + [{"event_id": "large"}] * 32
            )

        def __len__(self) -> int:
            return len(self.rows)

    def test_sampler_is_reproducible_capped_and_not_event_uniform(self) -> None:
        dataset = self.DummyDataset()
        first, first_info = build_capped_event_sampler(
            dataset,  # type: ignore[arg-type]
            alpha=0.5,
            min_weight=0.5,
            max_weight=4.0,
            seed=42,
        )
        second, second_info = build_capped_event_sampler(
            dataset,  # type: ignore[arg-type]
            alpha=0.5,
            min_weight=0.5,
            max_weight=4.0,
            seed=42,
        )
        first_draws = list(first)
        second_draws = list(second)
        self.assertEqual(first_draws, second_draws)
        self.assertEqual(first_info, second_info)
        for row in first_info["events"]:
            self.assertGreaterEqual(row["clipped_per_image_weight"], 0.5)
            self.assertLessEqual(row["clipped_per_image_weight"], 4.0)
        counts = Counter(dataset.rows[index]["event_id"] for index in first_draws)
        self.assertGreater(counts["large"], counts["tiny"])
        self.assertEqual(sum(counts.values()), len(dataset))


if __name__ == "__main__":
    unittest.main()
