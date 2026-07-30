#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CODE_ROOT="${CODE_ROOT:-${SCRIPT_DIR}}"
DEFAULT_DATASET_ROOT="/home/yr/code/datasets"
if [[ ! -d "${DEFAULT_DATASET_ROOT}" && -d "${SCRIPT_DIR}/../datasets" ]]; then
  DEFAULT_DATASET_ROOT="$(cd "${SCRIPT_DIR}/../datasets" && pwd)"
fi
export DATASET_ROOT="${DATASET_ROOT:-${DEFAULT_DATASET_ROOT}}"
export STAGE1_DATA_ROOT="${STAGE1_DATA_ROOT:-${DATASET_ROOT}/DisasterM3_stage1_optical_building}"
export DATA_ROOT="${DATA_ROOT:-${STAGE1_DATA_ROOT}}"
export PRACTICE_ROOT="${PRACTICE_ROOT:-${DATASET_ROOT}/DisasterM3_aria2_building_damage_practice}"
export STAGE2_DATA_ROOT="${STAGE2_DATA_ROOT:-${DATASET_ROOT}/DisasterM3_optical_sar_damage_minimal_v0.2}"
export OUT_ROOT="${OUT_ROOT:-${CODE_ROOT}/outputs}"
export STAGE2_OUT_ROOT="${STAGE2_OUT_ROOT:-${OUT_ROOT}/stage2}"
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
  echo "[INFO] STAGE2_OUT_ROOT=${STAGE2_OUT_ROOT}"
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

require_cuda() {
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi failed. Do not run formal Stage-2 work on CPU." >&2
    echo "[ERROR] See ../SERVER_RECOVERY.md for the diagnosed driver mismatch and recovery steps." >&2
    return 1
  fi
  python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("PyTorch CUDA is unavailable; refusing formal Stage-2 work")
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA devices:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"gpu[{index}]:", torch.cuda.get_device_name(index))
PY
}

stage2_v2_check_runtime() {
  show_paths
  [[ -d "${STAGE2_DATA_ROOT}" ]] || { echo "[ERROR] missing ${STAGE2_DATA_ROOT}" >&2; return 1; }
  for split in train val test; do
    [[ -f "${STAGE2_MANIFEST_DIR}/stage2_master_${split}.jsonl" ]] || {
      echo "[ERROR] missing stage2_master_${split}.jsonl" >&2
      return 1
    }
  done
  require_cuda
  echo "[OK] Stage-2 data, manifests, NVIDIA runtime, and sam3 PyTorch CUDA are available."
}

stage2_v2_strict_check_runtime() {
  show_paths
  local manifest_root="${STAGE2_V2_STRICT_MANIFEST_DIR:-${CODE_ROOT}/manifests/stage2_v2_clean_human_reviewed_20260624}"
  local audit_out="${STAGE2_V2_STRICT_AUDIT_OUT:-${STAGE2_OUT_ROOT}/v2_strict_v1_preflight/runtime_audit.json}"
  [[ -d "${STAGE2_DATA_ROOT}" ]] || { echo "[ERROR] missing ${STAGE2_DATA_ROOT}" >&2; return 1; }
  for split in train val test; do
    [[ -f "${manifest_root}/stage2_master_${split}.jsonl" ]] || {
      echo "[ERROR] missing strict stage2_master_${split}.jsonl in ${manifest_root}" >&2
      return 1
    }
  done
  python scripts/audit_stage2_v2_data.py \
    --data-root "${STAGE2_DATA_ROOT}" \
    --manifest-root "${manifest_root}" \
    --out "${audit_out}" \
    --require-event-disjoint \
    --skip-mask-stats
  require_cuda
  echo "[OK] strict-v1 manifests are content-unique, event-disjoint, and CUDA is available. audit=${audit_out}"
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

stage2_v2_audit_data() {
  show_paths
  python scripts/audit_stage2_v2_data.py \
    --data-root "${STAGE2_DATA_ROOT}" \
    --manifest-root "${STAGE2_MANIFEST_DIR}" \
    --out "${STAGE2_OUT_ROOT}/v2_preflight/data_audit.json"
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

stage2_rule_baselines() {
  show_paths
  python src/stage2/run_stage2_rule_baselines.py \
    --data-root "${STAGE2_DATA_ROOT}" \
    --manifest-root manifests \
    --split test \
    --output-dir "${STAGE2_OUT_ROOT}/rule_baselines"
}

stage2_visualize_dataloader() {
  show_paths
  python src/stage2/visualize_stage2.py \
    --config configs/stage2/b1_s2_predicted_prior_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --split train \
    --limit "${STAGE2_VIS_LIMIT:-30}" \
    --output-dir "${STAGE2_OUT_ROOT}/dataloader_visualization"
}

stage2_smoke_s1() {
  show_paths
  python src/stage2/train_stage2.py \
    --config configs/stage2/b1_s1_oracle_prior_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --output-dir "${STAGE2_OUT_ROOT}/debug_smoke_s1_oracle_prior" \
    --smoke-test \
    --no-pretrained \
    --num-workers 0
}

stage2_train_s0() {
  show_paths
  require_cuda
  python src/stage2/train_stage2.py \
    --config configs/stage2/b1_s0_no_prior_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --output-dir "${STAGE2_OUT_ROOT}/B1_S0_no_prior_unet_resnet34"
}

stage2_train_s1() {
  show_paths
  require_cuda
  python src/stage2/train_stage2.py \
    --config configs/stage2/b1_s1_oracle_prior_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --output-dir "${STAGE2_OUT_ROOT}/B1_S1_oracle_prior_unet_resnet34"
}

stage2_train_s2() {
  show_paths
  require_cuda
  python src/stage2/train_stage2.py \
    --config configs/stage2/b1_s2_predicted_prior_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --output-dir "${STAGE2_OUT_ROOT}/B1_S2_predicted_prior_unet_resnet34"
}

stage2_train_s3_oracle() {
  show_paths
  require_cuda
  python src/stage2/train_stage2.py \
    --config configs/stage2/b1_s3_oracle_prior_only_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --output-dir "${STAGE2_OUT_ROOT}/B1_S3_oracle_prior_only_unet_resnet34"
}

stage2_train_s3_predicted() {
  show_paths
  require_cuda
  python src/stage2/train_stage2.py \
    --config configs/stage2/b1_s3_predicted_prior_only_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --output-dir "${STAGE2_OUT_ROOT}/B1_S3_predicted_prior_only_unet_resnet34"
}

stage2_test_s0() {
  show_paths
  python src/stage2/test_stage2.py \
    --config configs/stage2/b1_s0_no_prior_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --checkpoint "${STAGE2_OUT_ROOT}/B1_S0_no_prior_unet_resnet34/checkpoints/best_metric.pth" \
    --split test \
    --output-dir "${STAGE2_OUT_ROOT}/B1_S0_no_prior_unet_resnet34/test"
}

stage2_test_s1() {
  show_paths
  python src/stage2/test_stage2.py \
    --config configs/stage2/b1_s1_oracle_prior_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --checkpoint "${STAGE2_OUT_ROOT}/B1_S1_oracle_prior_unet_resnet34/checkpoints/best_metric.pth" \
    --split test \
    --output-dir "${STAGE2_OUT_ROOT}/B1_S1_oracle_prior_unet_resnet34/test"
}

stage2_test_s2() {
  show_paths
  python src/stage2/test_stage2.py \
    --config configs/stage2/b1_s2_predicted_prior_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --checkpoint "${STAGE2_OUT_ROOT}/B1_S2_predicted_prior_unet_resnet34/checkpoints/best_metric.pth" \
    --split test \
    --output-dir "${STAGE2_OUT_ROOT}/B1_S2_predicted_prior_unet_resnet34/test"
}

stage2_test_s3_oracle() {
  show_paths
  python src/stage2/test_stage2.py \
    --config configs/stage2/b1_s3_oracle_prior_only_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --checkpoint "${STAGE2_OUT_ROOT}/B1_S3_oracle_prior_only_unet_resnet34/checkpoints/best_metric.pth" \
    --split test \
    --output-dir "${STAGE2_OUT_ROOT}/B1_S3_oracle_prior_only_unet_resnet34/test"
}

stage2_test_s3_predicted() {
  show_paths
  python src/stage2/test_stage2.py \
    --config configs/stage2/b1_s3_predicted_prior_only_unet_resnet34.yaml \
    --data-root "${STAGE2_DATA_ROOT}" \
    --checkpoint "${STAGE2_OUT_ROOT}/B1_S3_predicted_prior_only_unet_resnet34/checkpoints/best_metric.pth" \
    --split test \
    --output-dir "${STAGE2_OUT_ROOT}/B1_S3_predicted_prior_only_unet_resnet34/test"
}

stage2_analyze_prior_contribution() {
  show_paths
  python src/stage2/analyze_stage2_prior_contribution.py \
    --results-root "${STAGE2_OUT_ROOT}" \
    --out-dir "${STAGE2_OUT_ROOT}/prior_contribution_analysis"
}

stage2_v2_smoke() {
  show_paths
  require_cuda
  local config="${V2_CONFIG:-configs/stage2_v2/v2_a3_sar_predicted_prior.yaml}"
  local name="${V2_RUN_NAME:-v2_a3_smoke}"
  python src/stage2/train_stage2_v2.py \
    --config "${config}" \
    --data-root "${STAGE2_DATA_ROOT}" \
    --output-dir "${STAGE2_OUT_ROOT}/v2_smoke/${name}" \
    --smoke-test \
    --no-pretrained \
    --num-workers 0
}

stage2_v2_overfit() {
  show_paths
  require_cuda
  local config="${V2_CONFIG:-configs/stage2_v2/v2_a3_sar_predicted_prior.yaml}"
  local name="${V2_RUN_NAME:-v2_a3_overfit}"
  python src/stage2/train_stage2_v2.py \
    --config "${config}" \
    --data-root "${STAGE2_DATA_ROOT}" \
    --output-dir "${STAGE2_OUT_ROOT}/v2_overfit/${name}" \
    --overfit-batches 2 \
    --epochs "${V2_OVERFIT_EPOCHS:-30}" \
    --no-pretrained \
    --num-workers 0
}

stage2_v2_train() {
  show_paths
  require_cuda
  : "${V2_CONFIG:?set V2_CONFIG to a configs/stage2_v2 YAML path}"
  : "${V2_RUN_NAME:?set V2_RUN_NAME to a unique output directory name}"
  local seed="${V2_SEED:-42}"
  python src/stage2/train_stage2_v2.py \
    --config "${V2_CONFIG}" \
    --data-root "${STAGE2_DATA_ROOT}" \
    --output-dir "${STAGE2_OUT_ROOT}/v2_phase1/${V2_RUN_NAME}" \
    --seed "${seed}"
}

stage2_v2_test() {
  show_paths
  require_cuda
  : "${V2_CONFIG:?set V2_CONFIG to the run config}"
  : "${V2_CHECKPOINT:?set V2_CHECKPOINT to a v2 checkpoint}"
  : "${V2_TEST_NAME:?set V2_TEST_NAME to a unique output directory name}"
  python src/stage2/test_stage2_v2.py \
    --config "${V2_CONFIG}" \
    --data-root "${STAGE2_DATA_ROOT}" \
    --checkpoint "${V2_CHECKPOINT}" \
    --split "${V2_SPLIT:-test}" \
    --output-dir "${STAGE2_OUT_ROOT}/v2_evaluation/${V2_TEST_NAME}"
}

case "${1:-}" in
  check_env) check_env ;;
  stage2_v2_check_runtime) stage2_v2_check_runtime ;;
  stage2_v2_strict_check_runtime) stage2_v2_strict_check_runtime ;;
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
  stage2_v2_audit_data) stage2_v2_audit_data ;;
  smoke_stage2_minimal_dataloader) smoke_stage2_minimal_dataloader ;;
  verify_no_full_disasterm3_dependency) verify_no_full_disasterm3_dependency ;;
  stage2_rule_baselines) stage2_rule_baselines ;;
  stage2_visualize_dataloader) stage2_visualize_dataloader ;;
  stage2_smoke_s1) stage2_smoke_s1 ;;
  stage2_train_s0) stage2_train_s0 ;;
  stage2_train_s1) stage2_train_s1 ;;
  stage2_train_s2) stage2_train_s2 ;;
  stage2_train_s3_oracle) stage2_train_s3_oracle ;;
  stage2_train_s3_predicted) stage2_train_s3_predicted ;;
  stage2_test_s0) stage2_test_s0 ;;
  stage2_test_s1) stage2_test_s1 ;;
  stage2_test_s2) stage2_test_s2 ;;
  stage2_test_s3_oracle) stage2_test_s3_oracle ;;
  stage2_test_s3_predicted) stage2_test_s3_predicted ;;
  stage2_analyze_prior_contribution) stage2_analyze_prior_contribution ;;
  stage2_v2_smoke) stage2_v2_smoke ;;
  stage2_v2_overfit) stage2_v2_overfit ;;
  stage2_v2_train) stage2_v2_train ;;
  stage2_v2_test) stage2_v2_test ;;
  *)
    echo "Usage: bash instruction.sh {check_env|stage2_v2_check_runtime|stage2_v2_strict_check_runtime|stage2_v2_audit_data|stage2_v2_smoke|stage2_v2_overfit|stage2_v2_train|stage2_v2_test|...}"
    exit 1
    ;;
esac
