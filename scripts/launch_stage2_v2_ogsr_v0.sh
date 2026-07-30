#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/launch_stage2_v2_ogsr_v0.sh {d4|d5} GPU_INDEX [SEED]" >&2
  exit 2
fi

experiment_key="${1,,}"
gpu="$2"
seed="${3:-42}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GPU_INDEX must be an integer" >&2; exit 2; }
[[ "${seed}" =~ ^[0-9]+$ ]] || { echo "[ERROR] SEED must be an integer" >&2; exit 2; }

case "${experiment_key}" in
  d4)
    experiment="V2_OGSRV0_D4_pred_texture_residual_paired"
    config="configs/stage2_v2_ogsr_v0/v2_ogsr_d4_pred_texture_residual_paired.yaml"
    feature_root="outputs/stage2/ogsr_v0_features/pred_texture_paired_seed${seed}"
    diagnostic="pre+paired_sar+predicted_prior+OGSR_texture_residual"
    ;;
  d5)
    experiment="V2_OGSRV0_D5_pred_texture_residual_shuffled"
    config="configs/stage2_v2_ogsr_v0/v2_ogsr_d5_pred_texture_residual_shuffled.yaml"
    feature_root="outputs/stage2/ogsr_v0_features/pred_texture_shuffled_sar20260627_seed${seed}"
    diagnostic="pre+shuffled_sar+predicted_prior+OGSR_texture_residual"
    ;;
  d6)
    experiment="V2_OGSRV0_D6_pred_texture_plus_residual_paired"
    config="configs/stage2_v2_ogsr_v0/v2_ogsr_d6_pred_texture_plus_residual_paired.yaml"
    feature_root="outputs/stage2/ogsr_v0_features/pred_texture_paired_seed${seed}"
    diagnostic="pre+paired_sar+sar_texture+predicted_prior+OGSR_texture_residual"
    ;;
  d7)
    experiment="V2_OGSRV0_D7_pred_texture_plus_residual_shuffled"
    config="configs/stage2_v2_ogsr_v0/v2_ogsr_d7_pred_texture_plus_residual_shuffled.yaml"
    feature_root="outputs/stage2/ogsr_v0_features/pred_texture_shuffled_sar20260627_seed${seed}"
    diagnostic="pre+shuffled_sar+sar_texture+predicted_prior+OGSR_texture_residual"
    ;;
  *) echo "[ERROR] unknown OGSR-v0 Phase 2 experiment: ${experiment_key}" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
manifest_root="${repo_root}/manifests/stage2_v2_clean_human_reviewed_20260624"
python_bin="/home/yr/miniconda3/envs/sam3/bin/python"
timestamp="$(date +%Y%m%d_%H%M%S)"
session="s2ogsr-${experiment_key}-s${seed}-${timestamp}"
run_dir="${repo_root}/outputs/stage2/v2_ogsr_v0_phase2/${experiment}/seed_${seed}/run_${timestamp}"
log_path="${repo_root}/logs/stage2_v2_ogsr_v0_${experiment_key}_seed${seed}_${timestamp}.log"
audit_path="${repo_root}/outputs/stage2/ogsr_v0_preflight/audit_${experiment_key}_seed${seed}_${timestamp}.json"

[[ -x "${python_bin}" ]] || { echo "[ERROR] missing ${python_bin}" >&2; exit 1; }
[[ -d "${data_root}" ]] || { echo "[ERROR] missing data root: ${data_root}" >&2; exit 1; }
[[ -d "${manifest_root}" ]] || { echo "[ERROR] missing clean manifest root: ${manifest_root}" >&2; exit 1; }
[[ -f "${manifest_root}/overlap_review_summary.json" ]] || { echo "[ERROR] missing completed overlap review summary" >&2; exit 1; }
[[ -f "${repo_root}/${config}" ]] || { echo "[ERROR] missing config: ${config}" >&2; exit 1; }
[[ -f "${repo_root}/${feature_root}/completed.json" ]] || { echo "[ERROR] missing completed OGSR feature export: ${repo_root}/${feature_root}" >&2; exit 1; }
[[ -f "${repo_root}/${feature_root}/index_train.json" ]] || { echo "[ERROR] missing train feature index: ${repo_root}/${feature_root}/index_train.json" >&2; exit 1; }
[[ -f "${repo_root}/${feature_root}/index_val.json" ]] || { echo "[ERROR] missing val feature index: ${repo_root}/${feature_root}/index_val.json" >&2; exit 1; }
[[ ! -e "${run_dir}" ]] || { echo "[ERROR] output already exists: ${run_dir}" >&2; exit 1; }
tmux has-session -t "${session}" 2>/dev/null && { echo "[ERROR] tmux session exists: ${session}" >&2; exit 1; }

active_same_run="$(tmux list-sessions -F '#S' 2>/dev/null | grep -E "^s2ogsr-${experiment_key}-s${seed}-" || true)"
if [[ -n "${active_same_run}" ]]; then
  echo "[ERROR] the same OGSR-v0 experiment/seed is already active:" >&2
  echo "${active_same_run}" >&2
  exit 1
fi

gpu_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$' || true)"
if [[ -n "${gpu_pids}" && "${STAGE2_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "[ERROR] GPU ${gpu} already has compute processes: ${gpu_pids//$'\n'/, }" >&2
  echo "[ERROR] launch one Phase 2 process per GPU, or set STAGE2_ALLOW_BUSY_GPU=1 deliberately." >&2
  exit 1
fi

cd "${repo_root}"
env \
  PATH=/home/yr/miniconda3/envs/sam3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  STAGE2_V2_STRICT_MANIFEST_DIR="${manifest_root}" \
  STAGE2_V2_STRICT_AUDIT_OUT="${audit_path}" \
  bash instruction.sh stage2_v2_strict_check_runtime

echo "[PRELAUNCH] existing tmux sessions:"
tmux ls || true
echo "[PRELAUNCH] selected GPU:"
nvidia-smi -i "${gpu}" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo "[PRELAUNCH] protocol=stage2_v2_clean_human_reviewed_20260624"
echo "[PRELAUNCH] phase=OGSR-v0 Stage-2 damage head"
echo "[PRELAUNCH] diagnostic=${diagnostic}"
echo "[PRELAUNCH] audit=${audit_path}"
echo "[PRELAUNCH] config=${repo_root}/${config}"
echo "[PRELAUNCH] features=${repo_root}/${feature_root}"
echo "[PRELAUNCH] seed=${seed}"
echo "[PRELAUNCH] output=${run_dir}"
echo "[PRELAUNCH] log=${log_path}"

run_command="set -o pipefail; export CUDA_VISIBLE_DEVICES=${gpu}; cd \"${repo_root}\"; \"${python_bin}\" src/stage2/train_stage2_v2.py --config \"${config}\" --data-root \"${data_root}\" --output-dir \"${run_dir}\" --seed \"${seed}\" 2>&1 | tee \"${log_path}\""
tmux new-session -d -s "${session}" bash -lc "${run_command}"

echo "[STARTED] session=${session}"
echo "[STARTED] audit=${audit_path}"
echo "[STARTED] log=${log_path}"
echo "[STARTED] output=${run_dir}"
echo "[MONITOR] tmux attach -t ${session}"
echo "[MONITOR] tail -f '${log_path}'"
echo "[STOP] tmux kill-session -t ${session}"
