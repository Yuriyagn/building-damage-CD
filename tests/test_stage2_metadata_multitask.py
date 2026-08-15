from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for path in (SRC, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.metadata_multitask import (  # noqa: E402
    MetadataAwareUNet,
    diagnostic_shared_parameters,
    gradient_interaction,
)
from config import load_config  # noqa: E402
from scripts.build_stage2_metadata_shuffled_labels import (  # noqa: E402
    histogram_preserving_derangement,
)
from stage2.datasets import Stage2DamageDataset  # noqa: E402
from stage2.metadata_multitask import (  # noqa: E402
    DISASTER_CLASSES,
    DisasterClassificationMeter,
    encode_disaster_types,
    validate_disaster_classes,
)


class TinyEncoder(nn.Module):
    out_channels = (4, 8)

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(4, 8, 3, padding=1)
        self.layer4 = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        feature = torch.relu(self.layer4(torch.relu(self.stem(image))))
        return [image, feature]


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return torch.relu(self.block(features[-1]))


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyEncoder()
        self.decoder = TinyDecoder()
        self.segmentation_head = nn.Conv2d(8, 3, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.segmentation_head(self.decoder(self.encoder(image)))


class FourChannelDatasetTest(unittest.TestCase):
    def test_pre_sar_is_exact_rgb_plus_sar_without_prior_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(root / "images/pre.png")
            Image.fromarray(np.full((4, 4), 64, dtype=np.uint8)).save(root / "images/sar.png")
            Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(root / "masks/mask.png")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "event_id": "event",
                        "disaster_type": "earthquake",
                        "pre_image": "images/pre.png",
                        "post_sar": "images/sar.png",
                        "mask_multiclass": "masks/mask.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            item = Stage2DamageDataset(
                root, manifest, train=False, prior_type="none", input_mode="pre_sar"
            )[0]
        image = item["image"]
        self.assertEqual(tuple(image.shape), (4, 4, 4))
        self.assertTrue(torch.allclose(image[:3], torch.full((3, 4, 4), 128 / 255)))
        self.assertTrue(torch.allclose(image[3], torch.full((4, 4), 64 / 255)))
        self.assertEqual(item["disaster_type"], "earthquake")


class MetadataAwareModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.base = TinyUNet()
        self.wrapped = MetadataAwareUNet(
            copy.deepcopy(self.base), num_disaster_classes=len(DISASTER_CLASSES), dropout=0.0
        )
        self.image = torch.randn(2, 4, 16, 16)

    def test_damage_logits_match_unwrapped_model_at_initialization(self) -> None:
        expected = self.base(self.image)
        output = self.wrapped(self.image)
        self.assertEqual(tuple(output["damage_logits"].shape), (2, 3, 16, 16))
        self.assertEqual(tuple(output["disaster_logits"].shape), (2, 7))
        self.assertTrue(torch.equal(expected, output["damage_logits"]))

    def test_auxiliary_loss_reaches_shared_encoder_and_diagnostics_are_finite(self) -> None:
        output = self.wrapped(self.image)
        disaster_loss = nn.CrossEntropyLoss()(output["disaster_logits"], torch.tensor([0, 1]))
        damage_loss = output["damage_logits"].square().mean()
        diagnostics = gradient_interaction(
            damage_loss,
            disaster_loss,
            diagnostic_shared_parameters(self.wrapped),
            disaster_weight=0.1,
        )
        self.assertGreater(diagnostics["disaster_grad_norm"], 0.0)
        self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))
        disaster_loss.backward()
        self.assertGreater(float(self.wrapped.encoder.layer4.weight.grad.abs().sum()), 0.0)

    def test_checkpoint_roundtrip(self) -> None:
        expected = self.wrapped(self.image)["damage_logits"]
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "checkpoint.pth"
            torch.save(self.wrapped.state_dict(), path)
            restored = MetadataAwareUNet(TinyUNet(), num_disaster_classes=7, dropout=0.0)
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.assertTrue(torch.equal(expected, restored(self.image)["damage_logits"]))


class DisasterMetadataProtocolTest(unittest.TestCase):
    def test_m0_and_m1_change_only_auxiliary_weight_and_name(self) -> None:
        m0 = load_config(REPO_ROOT / "configs/stage2_metadata_multitask_v1/m0_rgb_sar.yaml")
        m1 = load_config(
            REPO_ROOT / "configs/stage2_metadata_multitask_v1/m1_rgb_sar_disaster_aux.yaml"
        )
        self.assertNotIn("test_manifest", m0["dataset"])
        self.assertNotIn("test_manifest", m1["dataset"])
        self.assertEqual(m0["dataset"]["input_mode"], "pre_sar")
        self.assertEqual(m0["model"], m1["model"])
        m0["experiment_name"] = m1["experiment_name"]
        m0["multitask"]["disaster_loss_weight"] = m1["multitask"]["disaster_loss_weight"]
        self.assertEqual(m0, m1)

    def test_frozen_mapping_and_unknown_label_failure(self) -> None:
        labels = encode_disaster_types(list(DISASTER_CLASSES))
        self.assertEqual(labels.tolist(), list(range(7)))
        self.assertEqual(validate_disaster_classes(list(DISASTER_CLASSES)), DISASTER_CLASSES)
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            encode_disaster_types(["tsunami"])
        with self.assertRaisesRegex(ValueError, "frozen protocol order"):
            validate_disaster_classes(list(reversed(DISASTER_CLASSES)))

    def test_classification_meter_reports_majority_control(self) -> None:
        meter = DisasterClassificationMeter()
        logits = torch.full((4, 7), -5.0)
        logits[:, 0] = 5.0
        target = torch.tensor([0, 0, 0, 1])
        meter.update(logits, target)
        metrics = meter.compute()
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["present_class_count"], 2.0)
        self.assertAlmostEqual(
            metrics["present_class_macro_f1"], metrics["majority_present_class_macro_f1"]
        )

    def test_shuffled_labels_are_fixed_deranged_and_histogram_preserving(self) -> None:
        rows = []
        for class_name, count in (("conflict", 4), ("earthquake", 3), ("fire", 2)):
            rows.extend(
                {"id": f"{class_name}-{index}", "disaster_type": class_name}
                for index in range(count)
            )
        first = histogram_preserving_derangement(rows)
        second = histogram_preserving_derangement(list(reversed(rows)))
        self.assertEqual(first, second)
        true = {str(row["id"]): str(row["disaster_type"]) for row in rows}
        self.assertTrue(all(first[sample_id] != label for sample_id, label in true.items()))
        self.assertEqual(Counter(first.values()), Counter(true.values()))


if __name__ == "__main__":
    unittest.main()
