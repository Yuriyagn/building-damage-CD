#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[01]$ ]]; then
  echo "Usage: bash scripts/run_stage2_v2_test_matrix.sh {0|1}" >&2
  exit 2
fi

gpu="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
python_bin="/home/yr/miniconda3/envs/sam3/bin/python"
runs_root="${repo_root}/outputs/stage2/v2_phase1"

if [[ "${gpu}" == "0" ]]; then
  experiments=(V2_A1_prior_only V2_A3_sar_predicted_prior)
else
  experiments=(V2_A2_sar_only V2_A4_shuffled_sar_predicted_prior)
fi
seeds=(42 3407 2026)

export CUDA_VISIBLE_DEVICES="${gpu}"
cd "${repo_root}"
for experiment in "${experiments[@]}"; do
  for seed in "${seeds[@]}"; do
    seed_dir="${runs_root}/${experiment}/seed_${seed}"
    shopt -s nullglob
    completed=("${seed_dir}"/run_*/completed.json)
    shopt -u nullglob
    if [[ ${#completed[@]} -ne 1 ]]; then
      echo "[ERROR] expected one completed run for ${experiment} seed ${seed}, found ${#completed[@]}" >&2
      exit 1
    fi
    run_dir="$(dirname "${completed[0]}")"
    output_dir="${run_dir}/test_best_grade"
    log_path="${run_dir}/eval_test_grade_20260621.log"
    if [[ -f "${output_dir}/metrics.json" ]]; then
      echo "[SKIP] completed ${output_dir}"
      continue
    fi
    if [[ -e "${output_dir}" ]]; then
      echo "[ERROR] incomplete evaluation output already exists: ${output_dir}" >&2
      exit 1
    fi
    echo "[EVAL] ${experiment} seed=${seed} gpu=${gpu}"
    "${python_bin}" src/stage2/test_stage2_v2.py \
      --config "${run_dir}/config_resolved.json" \
      --data-root "${data_root}" \
      --checkpoint "${run_dir}/checkpoints/best_bo_grade_macro_f1.pth" \
      --split test \
      --output-dir "${output_dir}" \
      --batch-size 16 \
      --num-workers 4 \
      --max-previews 8 \
      2>&1 | tee "${log_path}"
  done
done
echo "[COMPLETED] Stage-2 v2 test matrix shard GPU ${gpu}"
