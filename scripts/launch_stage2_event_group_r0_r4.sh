#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/launch_stage2_event_group_r0_r4.sh {r0|r4} {42|3407|2026} GPU_INDEX" >&2
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
  r0)
    experiment="R0_event_group_predicted_prior"
    config="configs/stage2_event_group_r0_r4_v1/r0_predicted_prior.yaml"
    manifest_rel="manifests/stage2_v2_event_group_v1_20260803"
    ;;
  r4)
    experiment="R4_bright_event_group_predicted_prior"
    config="configs/stage2_event_group_r0_r4_v1/r4_bright_predicted_prior.yaml"
    manifest_rel="manifests/stage2_v2_event_group_bright_r4_predicted_v1_20260803"
    ;;
  *) echo "[ERROR] unknown event-group comparison: ${experiment_key}" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
manifest_root="${repo_root}/${manifest_rel}"
python_bin="${PYTHON_BIN:-$(command -v python)}"
python_path="$(dirname "${python_bin}")"
timestamp="$(date +%Y%m%d_%H%M%S)"
session="s2egr-${experiment_key}-s${seed}-${timestamp}"
run_dir="${repo_root}/outputs/stage2/event_group_r0_r4_v1/${experiment}/seed_${seed}/run_${timestamp}"
log_path="${repo_root}/logs/stage2_event_group_${experiment_key}_seed${seed}_${timestamp}.log"
audit_path="${repo_root}/outputs/stage2/event_group_r0_r4_v1/preflight/audit_${experiment_key}_seed${seed}_${timestamp}.json"

[[ -x "${python_bin}" ]] || { echo "[ERROR] missing ${python_bin}" >&2; exit 1; }
[[ -d "${data_root}" ]] || { echo "[ERROR] missing data root: ${data_root}" >&2; exit 1; }
[[ -d "${manifest_root}" ]] || { echo "[ERROR] missing manifest root: ${manifest_root}" >&2; exit 1; }
[[ -f "${manifest_root}/predicted_prior/train.jsonl" ]] || { echo "[ERROR] predicted-prior manifest incomplete" >&2; exit 1; }
[[ -f "${repo_root}/${config}" ]] || { echo "[ERROR] missing config: ${config}" >&2; exit 1; }
[[ ! -e "${run_dir}" ]] || { echo "[ERROR] output already exists: ${run_dir}" >&2; exit 1; }

active_same_run="$(tmux list-sessions -F '#S' 2>/dev/null | grep -E "^s2egr-${experiment_key}-s${seed}-" || true)"
if [[ -n "${active_same_run}" ]]; then
  echo "[ERROR] the same event-group experiment/seed is already active:" >&2
  echo "${active_same_run}" >&2
  exit 1
fi

gpu_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$' || true)"
if [[ -n "${gpu_pids}" ]]; then
  echo "[ERROR] GPU ${gpu} already has compute processes: ${gpu_pids//$'\n'/, }" >&2
  exit 1
fi

cd "${repo_root}"
env \
  PATH="${python_path}:${PATH}" \
  STAGE2_V2_STRICT_MANIFEST_DIR="${manifest_root}" \
  STAGE2_V2_STRICT_AUDIT_OUT="${audit_path}" \
  bash instruction.sh stage2_v2_strict_check_runtime

echo "[PRELAUNCH] protocol=stage2_v2_event_group_r0_r4_v1"
echo "[PRELAUNCH] comparison=${experiment_key}"
echo "[PRELAUNCH] manifest_root=${manifest_root}"
echo "[PRELAUNCH] audit=${audit_path}"
echo "[PRELAUNCH] config=${repo_root}/${config}"
echo "[PRELAUNCH] seed=${seed}"
echo "[PRELAUNCH] output=${run_dir}"
echo "[PRELAUNCH] log=${log_path}"
nvidia-smi -i "${gpu}" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

run_command="set -o pipefail; export CUDA_VISIBLE_DEVICES=${gpu}; export STAGE2_DATA_AUDIT_PATH=\"${audit_path}\"; cd \"${repo_root}\"; \"${python_bin}\" src/stage2/train_stage2_v2.py --config \"${config}\" --data-root \"${data_root}\" --output-dir \"${run_dir}\" --seed \"${seed}\" 2>&1 | tee \"${log_path}\""
tmux new-session -d -s "${session}" bash -lc "${run_command}"

echo "[STARTED] session=${session}"
echo "[STARTED] audit=${audit_path}"
echo "[STARTED] log=${log_path}"
echo "[STARTED] output=${run_dir}"
echo "[MONITOR] tmux attach -t ${session}"
echo "[STOP] tmux kill-session -t ${session}"
