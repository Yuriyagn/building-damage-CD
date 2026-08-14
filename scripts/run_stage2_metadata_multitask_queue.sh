#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_stage2_metadata_multitask_queue.sh GPU_INDEX {m0_m1|m2}" >&2
  exit 2
fi

gpu="$1"
phase="${2,,}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GPU_INDEX must be an integer" >&2; exit 2; }
case "${phase}" in m0_m1|m2) ;; *) echo "[ERROR] phase must be m0_m1 or m2" >&2; exit 2 ;; esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
python_bin="${PYTHON_BIN:-$(command -v python)}"
audit_path="${repo_root}/outputs/stage2/metadata_multitask_v1/preflight/train_val_data_audit.json"
output_root="${repo_root}/outputs/stage2/metadata_multitask_v1"
mkdir -p "${output_root}" "${repo_root}/logs"

if [[ "${phase}" == "m0_m1" ]]; then
  conditions=(m0 m1)
else
  conditions=(m2)
fi

config_for() {
  case "$1" in
    m0) echo "configs/stage2_metadata_multitask_v1/m0_rgb_sar.yaml" ;;
    m1) echo "configs/stage2_metadata_multitask_v1/m1_rgb_sar_disaster_aux.yaml" ;;
    m2) echo "configs/stage2_metadata_multitask_v1/m2_rgb_sar_shuffled_disaster_aux.yaml" ;;
  esac
}

experiment_for() {
  case "$1" in
    m0) echo "M0_rgb_sar_damage_only" ;;
    m1) echo "M1_rgb_sar_disaster_aux" ;;
    m2) echo "M2_rgb_sar_shuffled_disaster_aux" ;;
  esac
}

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export STAGE2_DATA_ROOT="${data_root}"
export STAGE2_MANIFEST_DIR="${repo_root}/manifests/stage2_v2_event_group_bright_r4_predicted_v1_20260803"
export STAGE2_DATA_AUDIT_PATH="${audit_path}"

for seed in 42 3407 2026; do
  for condition in "${conditions[@]}"; do
    config="$(config_for "${condition}")"
    experiment="$(experiment_for "${condition}")"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    run_dir="${output_root}/${experiment}/seed_${seed}/run_${timestamp}"
    log_path="${repo_root}/logs/stage2_metadata_${condition}_seed${seed}_${timestamp}.log"
    echo "[QUEUE] condition=${condition} seed=${seed} gpu=${gpu} output=${run_dir}"
    "${python_bin}" src/stage2/train_stage2_v2.py \
      --config "${config}" \
      --data-root "${data_root}" \
      --output-dir "${run_dir}" \
      --seed "${seed}" 2>&1 | tee "${log_path}"
    [[ -f "${run_dir}/completed.json" ]] || {
      echo "[ERROR] training ended without completed.json: ${run_dir}" >&2
      exit 1
    }
    "${python_bin}" src/stage2/test_stage2_v2.py \
      --config "${config}" \
      --data-root "${data_root}" \
      --checkpoint "${run_dir}/checkpoints/best_bo_grade_macro_f1.pth" \
      --split val \
      --output-dir "${run_dir}/val_best_grade" \
      --num-workers 4 \
      --batch-size 4 \
      --max-previews 0 2>&1 | tee -a "${log_path}"
    [[ -f "${run_dir}/val_best_grade/metrics.json" ]] || {
      echo "[ERROR] validation ended without metrics.json: ${run_dir}" >&2
      exit 1
    }
    echo "[QUEUE] completed condition=${condition} seed=${seed} output=${run_dir}"
  done
done

echo "[QUEUE] phase=${phase} completed; test was not read or evaluated"
