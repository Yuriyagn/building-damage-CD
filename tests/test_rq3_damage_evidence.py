from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from summarize_rq3_damage_evidence import PILOT_GATES, evaluate  # noqa: E402
from validate_rq3_blind_evaluator_lock import validate_lock  # noqa: E402
from run_rq3_damage_evidence import (  # noqa: E402
    load_unlocked_development_amendment,
    require_anchor,
    resolve_evaluation_contract,
)
from recover_rq3_bright_georeference import (  # noqa: E402
    index_official_tiffs,
    recover_rows,
    sha256_grayscale_pixels,
)
from models.rq3_damage_evidence import fuse_control_sar, haar_dwt2, pool_component_features, rasterize_instance_logits  # noqa: E402
from stage2.rq3_components import predicted_prior_components  # noqa: E402
from stage2.rq3_runtime import reliability_auxiliary_loss, stage2_damage_loss, target_for_stage2_loss  # noqa: E402
from stage2.losses_v2 import BuildingOnlyGradeLoss  # noqa: E402
from stage2.datasets_v2 import Stage2V2Dataset  # noqa: E402
from stage2.numerical_integrity import (  # noqa: E402
    NumericalIntegrityError,
    checkpoint_integrity_sidecar,
    finite_python_values,
    guard_tensors,
    scan_tensors,
    strict_write_json,
    verify_checkpoint_sidecar,
)


def test_components_use_four_connectivity_and_mask_ambiguous_labels() -> None:
    prior = np.zeros((5, 5), dtype=np.float32)
    prior[0:2, 0:2] = 1.0
    prior[2:4, 2:4] = 1.0  # diagonal contact remains a separate component
    target = np.zeros((5, 5), dtype=np.uint8)
    target[0:2, 0:2] = 2
    target[2:4, 2:4] = np.asarray([[2, 3], [2, 3]])
    result = predicted_prior_components(prior, target, threshold=0.5, min_area=4, label_purity=0.8)
    assert result.component_count == 2
    assert result.valid_component_count == 1
    assert result.ambiguous_component_count == 1
    assert result.valid_loss_mask[0:2, 0:2].all()
    assert not result.valid_loss_mask[2:4, 2:4].any()


def test_false_positive_component_is_not_given_a_grade_label() -> None:
    prior = np.ones((3, 3), dtype=np.float32)
    target = np.zeros((3, 3), dtype=np.uint8)
    target[0, 0] = 2
    result = predicted_prior_components(prior, target, threshold=0.5, min_area=4, label_purity=0.8)
    assert result.component_count == 1
    assert result.valid_component_count == 0


def test_haar_shape_determinism_and_known_constant() -> None:
    image = torch.ones(2, 1, 5, 7)
    first = haar_dwt2(image)
    second = haar_dwt2(image)
    assert first.shape == (2, 4, 3, 4)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, 0], torch.full_like(first[:, 0], 2.0))
    torch.testing.assert_close(first[:, 1:], torch.zeros_like(first[:, 1:]))


def test_pool_and_rasterize_round_trip() -> None:
    features = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    components = torch.tensor([[[1, 1], [0, 2]]])
    pooled = pool_component_features(features, components)
    torch.testing.assert_close(pooled.pooled, torch.tensor([[2.0, 3.0], [7.0, 7.0]]))
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dense = rasterize_instance_logits(logits, pooled.pixel_index)
    torch.testing.assert_close(dense[0, :, 1, 0], torch.zeros(3))
    torch.testing.assert_close(dense[0, :, 1, 1], logits[1])


def test_fp16_large_component_pooling_matches_fp32_reference() -> None:
    features = torch.linspace(-128, 128, 512 * 512, dtype=torch.float32).reshape(
        1, 1, 512, 512
    ).half()
    components = torch.ones((1, 512, 512), dtype=torch.long)
    pooled = pool_component_features(features, components)
    expected = torch.tensor(
        [[features.float().mean().item(), features.float().max().item()]], dtype=torch.float32
    )
    assert pooled.pooled.dtype == torch.float32
    torch.testing.assert_close(pooled.pooled, expected, rtol=1e-6, atol=1e-5)


def test_fp16_multiple_large_components_and_empty_component_map() -> None:
    features = torch.ones((1, 2, 512, 512), dtype=torch.float16)
    components = torch.zeros((1, 512, 512), dtype=torch.long)
    components[:, :256] = 1
    components[:, 256:] = 2
    pooled = pool_component_features(features, components)
    assert pooled.pooled.shape == (2, 4)
    assert pooled.pooled.dtype == torch.float32
    assert torch.isfinite(pooled.pooled).all()
    empty = pool_component_features(features, torch.zeros_like(components))
    assert empty.pooled.shape == (0, 4)
    assert empty.pooled.dtype == torch.float32


def test_masked_target_and_reliability_supervision() -> None:
    target = torch.tensor([[[1, 2], [3, 1]]])
    valid = torch.tensor([[[True, False], [False, True]]])
    masked = target_for_stage2_loss({"rq3_loss_valid_mask": valid}, target)
    assert masked.tolist() == [[[1, 0], [0, 1]]]
    loss = reliability_auxiliary_loss(
        {"reliability_logits": torch.tensor([10.0, -10.0]), "instance_batch_index": torch.tensor([0, 1])},
        {"sar_is_paired": torch.tensor([True, False])}, torch.device("cpu"),
    )
    assert loss is not None and float(loss) < 0.001


def test_zero_reliability_exactly_reproduces_control_logits() -> None:
    control = torch.randn(2, 3, 4, 4)
    sar = torch.randn_like(control)
    assert torch.equal(fuse_control_sar(control, sar, torch.zeros(2, 1, 4, 4)), control)


def test_instance_loss_counts_each_component_once() -> None:
    instance_logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    pixel_index = torch.tensor([[[1, 1, 1, 2]]])
    target = torch.tensor([[[1, 1, 1, 2]]])
    output = {
        "instance_logits": instance_logits,
        "instance_pixel_index": pixel_index,
        "rq3_loss_valid_mask": torch.ones_like(target, dtype=torch.bool),
        "damage_logits": torch.zeros(1, 3, 1, 4),
    }
    observed = stage2_damage_loss(BuildingOnlyGradeLoss(), output, target)
    expected = torch.nn.functional.cross_entropy(instance_logits, torch.tensor([0, 1]))
    torch.testing.assert_close(observed, expected)


def test_instance_majority_vote_uses_int64_for_large_components() -> None:
    target = torch.cat(
        [
            torch.ones(3_000, dtype=torch.long),
            torch.full((40_000,), 2, dtype=torch.long),
            torch.full((5_000,), 3, dtype=torch.long),
        ]
    ).reshape(1, 1, -1)
    pixel_index = torch.ones_like(target)
    logits = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float16)
    output = {
        "instance_logits": logits,
        "instance_pixel_index": pixel_index,
        "rq3_loss_valid_mask": torch.ones_like(target, dtype=torch.bool),
        "damage_logits": torch.zeros(1, 3, 1, target.numel()),
    }
    observed = stage2_damage_loss(BuildingOnlyGradeLoss(), output, target)
    expected = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([1]))
    torch.testing.assert_close(observed.float(), expected, rtol=2e-3, atol=2e-3)


def test_numerical_integrity_rejects_nan_tensor_and_strict_json(tmp_path: Path) -> None:
    with pytest.raises(NumericalIntegrityError):
        guard_tensors(
            tmp_path,
            stage="unit_test_logits",
            items=[("logits", torch.tensor([0.0, float("nan")]))],
            step=7,
            batch_ids=["sample"],
        )
    failure = json.loads((tmp_path / "numerical_failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed_numerical"
    assert failure["first_nonfinite_tensor"] == "logits"
    assert finite_python_values({"metric": float("inf")}) == (False, "value.metric")
    with pytest.raises(ValueError):
        strict_write_json(tmp_path / "invalid.json", {"metric": float("nan")})


def test_checkpoint_sidecar_rejects_file_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    torch.save({"model_state": {"weight": torch.ones(2)}}, checkpoint)
    scan = scan_tensors([("checkpoint.model_state.weight", torch.ones(2))])
    checkpoint_integrity_sidecar(checkpoint, scan)
    assert verify_checkpoint_sidecar(checkpoint)["status"] == "passed"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        verify_checkpoint_sidecar(checkpoint)


def test_blind_lock_rejects_templates_and_accepts_complete_contract() -> None:
    base = {
        "protocol_id": "rq3_damage_evidence_decomposition_v1.0", "status": "accepted",
        "evaluator": "custodian", "accepted_at": "2026-08-20T00:00:00Z",
        "written_confirmation_sha256": "1" * 64, "event_count": 5,
        "event_ids_sha256": "2" * 64, "input_manifest_sha256": "3" * 64,
        "hidden_labels_sha256": "4" * 64, "metric_script_sha256": "5" * 64,
        "labels_withheld": True, "pixel_semantic_evaluation": True,
        "sensor_metadata_available": True, "damaged_event_count": 2,
        "destroyed_event_count": 2, "geospatial_overlap_audit": "passed",
        "excluded_event_overlap": False, "california_2025_excluded": True,
        "jamaica_2025_excluded": True, "format_dry_run_limit": 1,
        "scored_submission_limit": 1,
        "resubmission_policy": "only_if_no_score_was_computed_or_disclosed",
    }
    assert validate_lock(base) == []
    base["hidden_labels_sha256"] = "0" * 64
    assert any("zero digest" in error for error in validate_lock(base))


def test_pilot_gate_is_all_or_nothing() -> None:
    metrics = {name: threshold for name, (_, threshold) in PILOT_GATES.items()}
    result = evaluate({"phase": "pilot", "experiment": "E1", "selected_factors": ["instance"], "metrics": metrics})
    assert result["passed"] is True
    metrics["destroyed_delta"] = -0.011
    assert evaluate({"phase": "pilot", "metrics": metrics})["passed"] is False


def test_c3_and_reliability_mix_move_metadata_with_sar(tmp_path: Path) -> None:
    rows = []
    for index, value in enumerate((20, 220)):
        Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(tmp_path / f"sar{index}.png")
        Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(tmp_path / f"mask{index}.png")
        np.savez_compressed(tmp_path / f"prior{index}.npz", prob=np.ones((4, 4), dtype=np.float32))
        rows.append({
            "id": f"s{index}", "event_id": "event", "post_sar": f"sar{index}.png",
            "mask_multiclass": f"mask{index}.png", "building_prior": f"prior{index}.npz",
            "sar_provenance": {"provider": f"provider{index}", "mode": "spot", "polarization": "VV"},
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    permutation = tmp_path / "permutation.json"
    permutation.write_text(json.dumps({"mapping": {"s0": "s1", "s1": "s0"}}), encoding="utf-8")

    deranged = Stage2V2Dataset(
        data_root=tmp_path, manifest=manifest, train=False, input_mode="sar_prior",
        sar_permutation_file=permutation,
    )
    item = deranged[0]
    assert item["sar_source_id"] == "s1"
    assert item["sar_provider"] == "provider1"
    assert float(item["sar"].mean()) == pytest.approx(220 / 255)

    mixed = Stage2V2Dataset(
        data_root=tmp_path, manifest=manifest, train=True, input_mode="sar_prior",
        rq3_reliability_permutation_file=permutation,
    )
    assert len(mixed) == 4
    assert mixed[0]["sar_is_paired"] is True
    assert mixed[1]["sar_is_paired"] is False
    assert mixed[1]["sar_provider"] == "provider1"


def test_sensor_normalization_falls_back_to_provider_then_global(tmp_path: Path) -> None:
    Image.fromarray(np.full((4, 4), 128, dtype=np.uint8)).save(tmp_path / "sar.png")
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(tmp_path / "mask.png")
    np.savez_compressed(tmp_path / "prior.npz", prob=np.ones((4, 4), dtype=np.float32))
    row = {"id": "s", "event_id": "e", "post_sar": "sar.png", "mask_multiclass": "mask.png", "building_prior": "prior.npz", "sar_provenance": {"provider": "p", "mode": "unknown", "polarization": "unknown"}}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "schema_version": "rq3_sar_calibration_v1", "groups": {},
        "providers": {"p": {"median": 128 / 255, "mad": 0.1}},
        "global": {"median": 0.0, "mad": 1.0},
    }), encoding="utf-8")
    dataset = Stage2V2Dataset(data_root=tmp_path, manifest=manifest, train=False, input_mode="sar_prior", sar_calibration_file=calibration)
    assert float(dataset[0]["sar"].abs().max()) < 1e-5


def test_full_phase_requires_its_passed_pilot_stack(tmp_path: Path) -> None:
    summary = tmp_path / "pilot.json"
    summary.write_text(json.dumps({
        "protocol_id": "rq3_damage_evidence_decomposition_v1.0",
        "phase": "pilot", "experiment": "E1", "passed": True,
        "selected_factors": ["instance"],
    }), encoding="utf-8")
    assert require_anchor(summary, "E1", "full") == ["instance"]
    with pytest.raises(ValueError):
        require_anchor(None, "E1", "full")


def test_unlocked_development_amendment_is_explicit_and_claim_limited(tmp_path: Path) -> None:
    amendment = tmp_path / "amendment.yaml"
    amendment.write_text(
        """
amendment_id: rq3_damage_evidence_decomposition_unlocked_dev_v1.0
parent_protocol_id: rq3_damage_evidence_decomposition_v1.0
authority: user_explicit_instruction
decision: waive_blind_evaluator_lock_for_exposed_development
execution_class: exposed_development_only
allowed_phases: [pilot, full]
blind_evaluator_lock_required: false
claim_policy:
  development_screening_evidence: allowed
  independent_blind_confirmation: forbidden
  unseen_event_robustness: forbidden
  deployment_readiness: forbidden
""".strip() + "\n",
        encoding="utf-8",
    )
    payload = load_unlocked_development_amendment(amendment, "pilot")
    assert payload["execution_class"] == "exposed_development_only"
    contract = resolve_evaluation_contract(
        phase="pilot", blind_evaluator_lock=None,
        unlocked_development_amendment=amendment,
    )
    assert contract["status"] == "waived_by_explicit_user_amendment"
    assert contract["claim_policy"]["unseen_event_robustness"] == "forbidden"


def test_formal_rq3_still_rejects_missing_or_conflicting_contracts(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires either"):
        resolve_evaluation_contract(
            phase="pilot", blind_evaluator_lock=None,
            unlocked_development_amendment=None,
        )
    lock = tmp_path / "lock.json"
    amendment = tmp_path / "amendment.yaml"
    with pytest.raises(RuntimeError, match="not both"):
        resolve_evaluation_contract(
            phase="pilot", blind_evaluator_lock=lock,
            unlocked_development_amendment=amendment,
        )


def test_bright_georeference_recovery_is_exact_and_fail_closed(tmp_path: Path) -> None:
    official = tmp_path / "official"
    official.mkdir()

    def write_geotiff(path: Path, value: int, left: float) -> None:
        import rasterio
        from rasterio.transform import from_origin

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(left, 10.0, 0.1, 0.1),
        ) as dataset:
            dataset.write(np.full((1, 4, 4), value, dtype=np.uint8))

    write_geotiff(official / "unique.tif", 10, 1.0)
    write_geotiff(official / "duplicate_a.tif", 20, 2.0)
    (official / "duplicate_b.tif").write_bytes((official / "duplicate_a.tif").read_bytes())
    unmatched = tmp_path / "unmatched.tif"
    write_geotiff(unmatched, 30, 3.0)

    index = index_official_tiffs(official, workers=2)
    sidecar_rows = [
        {"id": "u", "event_id": "e", "post_sar": str(official / "unique.tif")},
        {"id": "a", "event_id": "e", "post_sar": str(official / "duplicate_a.tif")},
        {"id": "x", "event_id": "e", "post_sar": str(unmatched)},
    ]
    recovered = {row["sample_id"]: row for row in recover_rows(sidecar_rows, index, official)}
    assert recovered["u"]["content_match_status"] == "unique"
    assert recovered["u"]["georeference_status"] == "recovered"
    assert recovered["u"]["official_geotiff"]["epsg"] == 4326
    assert recovered["a"]["content_match_status"] == "ambiguous"
    assert recovered["a"]["georeference_status"] == "not_recovered"
    assert "official_relative_path" not in recovered["a"]
    assert recovered["x"]["content_match_status"] == "unmatched"


def test_pixel_hash_ignores_container_metadata_but_file_hash_does_not(tmp_path: Path) -> None:
    from recover_rq3_bright_georeference import sha256_file

    pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    Image.fromarray(pixels).save(first, compression="raw")
    Image.fromarray(pixels).save(second, compression="tiff_lzw")
    assert sha256_file(first) != sha256_file(second)
    assert sha256_grayscale_pixels(first) == sha256_grayscale_pixels(second)
