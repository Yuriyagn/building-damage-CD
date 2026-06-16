#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CODE_ROOT="${CODE_ROOT:-${SCRIPT_DIR}}"
export STAGE1_DATA_ROOT="${STAGE1_DATA_ROOT:-/home/yr/code/datasets/DisasterM3_stage1_optical_building}"
export DATA_ROOT="${DATA_ROOT:-${STAGE1_DATA_ROOT}}"
export DATASET_ROOT="${DATASET_ROOT:-/home/yr/code/datasets}"
export PRACTICE_ROOT="${PRACTICE_ROOT:-${DATASET_ROOT}/DisasterM3_aria2_building_damage_practice}"
export STAGE2_DATA_ROOT="${STAGE2_DATA_ROOT:-${DATASET_ROOT}/DisasterM3_optical_sar_damage_minimal_v0.2}"
export OUT_ROOT="${OUT_ROOT:-${CODE_ROOT}/outputs}"
export STAGE1_OFFICIAL_DIR="${STAGE1_OFFICIAL_DIR:-${PRACTICE_ROOT}/stage1_official/O1_unet_resnet34_freq}"
export SOURCE_STAGE1_PRIOR_DIR="${SOURCE_STAGE1_PRIOR_DIR:-${PRACTICE_ROOT}/derived_priors/O1_unet_resnet34_freq}"
export MINIMAL_STAGE1_PRIOR_DIR="${MINIMAL_STAGE1_PRIOR_DIR:-${STAGE2_DATA_ROOT}/building_priors/O1_unet_resnet34_freq}"
export STAGE1_PRIOR_DIR="${STAGE1_PRIOR_DIR:-${MINIMAL_STAGE1_PRIOR_DIR}}"
export STAGE2_MANIFEST_DIR="${STAGE2_MANIFEST_DIR:-${STAGE2_DATA_ROOT}/manifests}"

cd "${CODE_ROOT}"
mkdir -p "${OUT_ROOT}" logs

TRAIN_EXTRA_ARGS=()
if [[ -n "${EARLY_STOPPING_PATIENCE:-}" ]]; then
  TRAIN_EXTRA_ARGS+=(--early-stopping-patience "${EARLY_STOPPING_PATIENCE}")
fi
if [[ -n "${EARLY_STOPPING_MIN_DELTA:-}" ]]; then
  TRAIN_EXTRA_ARGS+=(--early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}")
fi
if [[ -n "${EARLY_STOPPING_MIN_EPOCHS:-}" ]]; then
  TRAIN_EXTRA_ARGS+=(--early-stopping-min-epochs "${EARLY_STOPPING_MIN_EPOCHS}")
fi

show_paths() {
  echo "[INFO] CODE_ROOT=${CODE_ROOT}"
  echo "[INFO] STAGE1_DATA_ROOT=${STAGE1_DATA_ROOT}"
  echo "[INFO] DATA_ROOT=${DATA_ROOT}"
  echo "[INFO] DATASET_ROOT=${DATASET_ROOT}"
  echo "[INFO] PRACTICE_ROOT=${PRACTICE_ROOT}"
  echo "[INFO] STAGE2_DATA_ROOT=${STAGE2_DATA_ROOT}"
  echo "[INFO] SOURCE_STAGE1_PRIOR_DIR=${SOURCE_STAGE1_PRIOR_DIR}"
  echo "[INFO] MINIMAL_STAGE1_PRIOR_DIR=${MINIMAL_STAGE1_PRIOR_DIR}"
  echo "[INFO] STAGE2_MANIFEST_DIR=${STAGE2_MANIFEST_DIR}"
  echo "[INFO] OUT_ROOT=${OUT_ROOT}"
  if (( ${#TRAIN_EXTRA_ARGS[@]} )); then
    echo "[INFO] TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS[*]}"
  fi
}

check_env() {
  show_paths
  python - <<'PY'
import importlib.util
import sys
import torch

print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
if importlib.util.find_spec("segmentation_models_pytorch") is None:
    raise SystemExit("segmentation_models_pytorch is not installed")
import segmentation_models_pytorch as smp
print("smp:", getattr(smp, "__version__", "installed"))
PY
}

check_data() {
  show_paths
  python tools/check_stage1_dataset.py \
    --data-root "${DATA_ROOT}" \
    --manifest "${DATA_ROOT}/manifests/building_extract_train_sar_qc_freq.jsonl" \
    --check-mask-values
  python tools/check_stage1_dataset.py \
    --data-root "${DATA_ROOT}" \
    --manifest "${DATA_ROOT}/manifests/building_extract_val_sar_qc.jsonl"
  python tools/check_stage1_dataset.py \
    --data-root "${DATA_ROOT}" \
    --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc.jsonl"
}

smoke_unet_freq() {
  python src/train.py \
    --config configs/stage1_optical_building_unet_resnet34_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/debug_smoke_unet_freq" \
    --smoke-test
}

overfit_unet_freq() {
  python src/train.py \
    --config configs/stage1_optical_building_unet_resnet34_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/debug_overfit_unet_freq" \
    --overfit-batches 2 \
    --epochs 30
}

train_unet_freq() {
  python src/train.py \
    --config configs/stage1_optical_building_unet_resnet34_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/O1_unet_resnet34_freq" \
    "${TRAIN_EXTRA_ARGS[@]}"
}

test_unet_freq() {
  local ckpt="${OUT_ROOT}/O1_unet_resnet34_freq/checkpoints/best_iou.pth"
  python src/test.py --config configs/stage1_optical_building_unet_resnet34_freq.yaml --data-root "${DATA_ROOT}" --checkpoint "${ckpt}" --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc.jsonl" --output-dir "${OUT_ROOT}/O1_unet_resnet34_freq/test_all"
  python src/test.py --config configs/stage1_optical_building_unet_resnet34_freq.yaml --data-root "${DATA_ROOT}" --checkpoint "${ckpt}" --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc_freq.jsonl" --output-dir "${OUT_ROOT}/O1_unet_resnet34_freq/test_freq"
  python src/test.py --config configs/stage1_optical_building_unet_resnet34_freq.yaml --data-root "${DATA_ROOT}" --checkpoint "${ckpt}" --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc_rare.jsonl" --output-dir "${OUT_ROOT}/O1_unet_resnet34_freq/test_rare"
  python src/test.py --config configs/stage1_optical_building_unet_resnet34_freq.yaml --data-root "${DATA_ROOT}" --checkpoint "${ckpt}" --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc_ok.jsonl" --output-dir "${OUT_ROOT}/O1_unet_resnet34_freq/test_ok"
}

train_unet_all() {
  python src/train.py \
    --config configs/stage1_optical_building_unet_resnet34_all.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/O2_unet_resnet34_all" \
    "${TRAIN_EXTRA_ARGS[@]}"
}

test_unet_all() {
  python src/test.py \
    --config configs/stage1_optical_building_unet_resnet34_all.yaml \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${OUT_ROOT}/O2_unet_resnet34_all/checkpoints/best_iou.pth" \
    --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc.jsonl" \
    --output-dir "${OUT_ROOT}/O2_unet_resnet34_all/test_all"
}

train_deeplab_freq() {
  python src/train.py \
    --config configs/stage1_optical_building_deeplabv3p_resnet50_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/O3_deeplabv3p_resnet50_freq" \
    "${TRAIN_EXTRA_ARGS[@]}"
}

test_deeplab_freq() {
  python src/test.py \
    --config configs/stage1_optical_building_deeplabv3p_resnet50_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${OUT_ROOT}/O3_deeplabv3p_resnet50_freq/checkpoints/best_iou.pth" \
    --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc.jsonl" \
    --output-dir "${OUT_ROOT}/O3_deeplabv3p_resnet50_freq/test_all"
}

train_segformer_freq() {
  python src/train.py \
    --config configs/stage1_optical_building_segformer_b0_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/O4_segformer_b0_freq" \
    "${TRAIN_EXTRA_ARGS[@]}"
}

test_segformer_freq() {
  python src/test.py \
    --config configs/stage1_optical_building_segformer_b0_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${OUT_ROOT}/O4_segformer_b0_freq/checkpoints/best_iou.pth" \
    --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc.jsonl" \
    --output-dir "${OUT_ROOT}/O4_segformer_b0_freq/test_all"
}

summarize() {
  python tools/summarize_stage1_results.py \
    --outputs-root "${OUT_ROOT}" \
    --out "${OUT_ROOT}/stage1_summary.csv"
}

freeze_stage1_o1() {
  show_paths
  python scripts/freeze_stage1_o1.py \
    --exp-dir "${OUT_ROOT}/O1_unet_resnet34_freq" \
    --practice-root "${PRACTICE_ROOT}" \
    --out "${STAGE1_OFFICIAL_DIR}" \
    --config configs/stage1_optical_building_unet_resnet34_freq.yaml
}

tune_stage1_o1_threshold() {
  show_paths
  python scripts/tune_stage1_threshold.py \
    --config configs/stage1_optical_building_unet_resnet34_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${OUT_ROOT}/O1_unet_resnet34_freq/checkpoints/best_iou.pth" \
    --manifest "${DATA_ROOT}/manifests/building_extract_val_sar_qc.jsonl" \
    --out-dir "${STAGE1_OFFICIAL_DIR}"
}

export_stage1_o1_priors_source() {
  show_paths
  python scripts/export_stage1_o1_priors.py \
    --config configs/stage1_optical_building_unet_resnet34_freq.yaml \
    --checkpoint "${OUT_ROOT}/O1_unet_resnet34_freq/checkpoints/best_iou.pth" \
    --threshold-json "${STAGE1_OFFICIAL_DIR}/official_threshold.json" \
    --practice-root "${PRACTICE_ROOT}" \
    --data-root "${DATASET_ROOT}" \
    --input-manifests \
      splits/v0.2_qc/building_mask_train_sar_noempty_qc.jsonl \
      splits/v0.2_qc/building_mask_val_sar_qc.jsonl \
      splits/v0.2_qc/building_mask_test_sar_qc.jsonl \
    --split-names train val test \
    --out-dir "${SOURCE_STAGE1_PRIOR_DIR}"
}

build_minimal_stage2_package() {
  show_paths
  python scripts/build_minimal_optical_sar_damage_package.py \
    --practice-root "${PRACTICE_ROOT}" \
    --source-data-root "${DATASET_ROOT}" \
    --stage1-prior-dir "${SOURCE_STAGE1_PRIOR_DIR}" \
    --stage1-binary-dir "${PRACTICE_ROOT}/derived_masks_stage1_building" \
    --threshold-json "${STAGE1_OFFICIAL_DIR}/official_threshold.json" \
    --out-dir "${STAGE2_DATA_ROOT}"
}

export_stage1_o1_priors() {
  show_paths
  python scripts/export_stage1_o1_priors.py \
    --config configs/stage1_optical_building_unet_resnet34_freq.yaml \
    --checkpoint "${OUT_ROOT}/O1_unet_resnet34_freq/checkpoints/best_iou.pth" \
    --threshold-json "${STAGE1_OFFICIAL_DIR}/official_threshold.json" \
    --practice-root "${STAGE2_DATA_ROOT}" \
    --data-root "${STAGE2_DATA_ROOT}" \
    --input-manifests \
      manifests/stage1_prior_input_train.jsonl \
      manifests/stage1_prior_input_val.jsonl \
      manifests/stage1_prior_input_test.jsonl \
    --split-names train val test \
    --out-dir "${MINIMAL_STAGE1_PRIOR_DIR}"
}

prepare_stage2_manifests() {
  build_minimal_stage2_package
}

check_stage2_manifests() {
  show_paths
  python scripts/check_stage2_sar_damage_manifests.py \
    --manifest-root "${STAGE2_MANIFEST_DIR}" \
    --data-root "${STAGE2_DATA_ROOT}" \
    --out "${STAGE2_DATA_ROOT}/reports/stage2_manifest_check.json"
}

smoke_stage2_minimal_dataloader() {
  show_paths
  python scripts/smoke_stage2_minimal_dataloader.py \
    --data-root "${STAGE2_DATA_ROOT}" \
    --manifest-root "${STAGE2_MANIFEST_DIR}" \
    --out "${STAGE2_DATA_ROOT}/reports/stage2_dataloader_smoke.json"
}

verify_no_full_disasterm3_dependency() {
  local failed=0
  local prefix="DisasterM3_aria2/DisasterM3_"
  for suffix in "Instruct" "Bench"; do
    local pattern="${prefix}${suffix}"
    if grep -R "${pattern}" -n scripts src tools instruction.sh 2>/dev/null; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "full DisasterM3 path dependency remains"
    return 1
  fi
  echo "No full DisasterM3_Instruct/Bench path dependency found in scripts/src/tools/instruction.sh"
}

case "${1:-}" in
  check_env) check_env ;;
  check_data) check_data ;;
  smoke_unet_freq) smoke_unet_freq ;;
  overfit_unet_freq) overfit_unet_freq ;;
  train_unet_freq) train_unet_freq ;;
  test_unet_freq) test_unet_freq ;;
  train_unet_all) train_unet_all ;;
  test_unet_all) test_unet_all ;;
  train_deeplab_freq) train_deeplab_freq ;;
  test_deeplab_freq) test_deeplab_freq ;;
  train_segformer_freq) train_segformer_freq ;;
  test_segformer_freq) test_segformer_freq ;;
  summarize) summarize ;;
  freeze_stage1_o1) freeze_stage1_o1 ;;
  tune_stage1_o1_threshold) tune_stage1_o1_threshold ;;
  export_stage1_o1_priors_source) export_stage1_o1_priors_source ;;
  build_minimal_stage2_package) build_minimal_stage2_package ;;
  export_stage1_o1_priors) export_stage1_o1_priors ;;
  prepare_stage2_manifests) prepare_stage2_manifests ;;
  check_stage2_manifests) check_stage2_manifests ;;
  smoke_stage2_minimal_dataloader) smoke_stage2_minimal_dataloader ;;
  verify_no_full_disasterm3_dependency) verify_no_full_disasterm3_dependency ;;
  *)
    echo "Usage: bash instruction.sh {check_env|check_data|smoke_unet_freq|overfit_unet_freq|train_unet_freq|test_unet_freq|train_unet_all|test_unet_all|train_deeplab_freq|test_deeplab_freq|train_segformer_freq|test_segformer_freq|summarize|freeze_stage1_o1|tune_stage1_o1_threshold|export_stage1_o1_priors_source|build_minimal_stage2_package|export_stage1_o1_priors|prepare_stage2_manifests|check_stage2_manifests|smoke_stage2_minimal_dataloader|verify_no_full_disasterm3_dependency}"
    exit 1
    ;;
esac
