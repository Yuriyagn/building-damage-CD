#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/launch_stage2_v2_strict_run.sh {a1|a2|a3|a4} {42|3407|2026} GPU_INDEX" >&2
  exit 2
fi

experiment_key="${1,,}"
seed="$2"
gpu="$3"
case "${seed}" in
  42|3407|2026) ;;
  *) echo "[ERROR] formal seed must be 42, 3407, or 2026" >&2; exit 2 ;;
esac
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GPU_INDEX must be an integer" >&2; exit 2; }

case "${experiment_key}" in
  a1)
    experiment="V2S_A1_prior_only"
    config="configs/stage2_v2_strict_v1/v2s_a1_prior_only.yaml"
    ;;
  a2)
    experiment="V2S_A2_sar_only"
    config="configs/stage2_v2_strict_v1/v2s_a2_sar_only.yaml"
    ;;
  a3)
    experiment="V2S_A3_sar_predicted_prior"
    config="configs/stage2_v2_strict_v1/v2s_a3_sar_predicted_prior.yaml"
    ;;
  a4)
    experiment="V2S_A4_shuffled_sar_predicted_prior"
    config="configs/stage2_v2_strict_v1/v2s_a4_shuffled_sar_predicted_prior.yaml"
    ;;
  *) echo "[ERROR] unknown strict-v1 experiment: ${experiment_key}" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
manifest_root="${repo_root}/manifests/stage2_v2_clean_human_reviewed_20260624"
python_bin="/home/yr/miniconda3/envs/sam3/bin/python"
timestamp="$(date +%Y%m%d_%H%M%S)"
session="s2v2s-${experiment_key}-s${seed}-${timestamp}"
run_dir="${repo_root}/outputs/stage2/v2_strict_v1_phase1/${experiment}/seed_${seed}/run_${timestamp}"
log_path="${repo_root}/logs/stage2_v2_strict_${experiment_key}_seed${seed}_${timestamp}.log"
audit_path="${repo_root}/outputs/stage2/v2_strict_v1_preflight/audit_${experiment_key}_seed${seed}_${timestamp}.json"

[[ -x "${python_bin}" ]] || { echo "[ERROR] missing ${python_bin}" >&2; exit 1; }
[[ -d "${data_root}" ]] || { echo "[ERROR] missing data root: ${data_root}" >&2; exit 1; }
[[ -d "${manifest_root}" ]] || { echo "[ERROR] missing strict manifest root: ${manifest_root}" >&2; exit 1; }
[[ -f "${manifest_root}/overlap_review_summary.json" ]] || { echo "[ERROR] missing completed overlap review summary" >&2; exit 1; }
[[ -f "${repo_root}/${config}" ]] || { echo "[ERROR] missing config: ${config}" >&2; exit 1; }
[[ ! -e "${run_dir}" ]] || { echo "[ERROR] output already exists: ${run_dir}" >&2; exit 1; }
tmux has-session -t "${session}" 2>/dev/null && { echo "[ERROR] tmux session exists: ${session}" >&2; exit 1; }

active_same_run="$(tmux list-sessions -F '#S' 2>/dev/null | grep -E "^s2v2s-${experiment_key}-s${seed}-" || true)"
if [[ -n "${active_same_run}" ]]; then
  echo "[ERROR] the same strict-v1 experiment/seed is already active:" >&2
  echo "${active_same_run}" >&2
  exit 1
fi

gpu_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$' || true)"
if [[ -n "${gpu_pids}" && "${STAGE2_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "[ERROR] GPU ${gpu} already has compute processes: ${gpu_pids//$'\n'/, }" >&2
  echo "[ERROR] strict-v1 formal runs require one training process per GPU." >&2
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
echo "[PRELAUNCH] audit=${audit_path}"
echo "[PRELAUNCH] config=${repo_root}/${config}"
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
