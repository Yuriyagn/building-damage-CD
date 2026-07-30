#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash scripts/export_stage2_ogsr_v0_features.sh {paired|shuffled} GPU_INDEX PREDICTOR_RUN_DIR [SEED]" >&2
  exit 2
fi

feature_key="${1,,}"
gpu="$2"
predictor_run_dir="$(realpath "$3")"
seed="${4:-42}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GPU_INDEX must be an integer" >&2; exit 2; }
[[ "${seed}" =~ ^[0-9]+$ ]] || { echo "[ERROR] SEED must be an integer" >&2; exit 2; }

case "${feature_key}" in
  paired)
    sar_shuffle_mode="paired"
    feature_root_name="pred_texture_paired_seed${seed}"
    ;;
  shuffled)
    sar_shuffle_mode="within_event"
    feature_root_name="pred_texture_shuffled_sar20260627_seed${seed}"
    ;;
  *) echo "[ERROR] unknown OGSR feature export key: ${feature_key}" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
manifest_root="${repo_root}/manifests/stage2_v2_clean_human_reviewed_20260624"
python_bin="/home/yr/miniconda3/envs/sam3/bin/python"
config="${predictor_run_dir}/config_resolved.json"
checkpoint="${predictor_run_dir}/checkpoints/best_val_loss.pth"
output_dir="${repo_root}/outputs/stage2/ogsr_v0_features/${feature_root_name}"
timestamp="$(date +%Y%m%d_%H%M%S)"
session="s2ogsrfeat-${feature_key}-s${seed}-${timestamp}"
log_path="${repo_root}/logs/stage2_ogsr_v0_features_${feature_key}_seed${seed}_${timestamp}.log"
audit_path="${repo_root}/outputs/stage2/ogsr_v0_preflight/audit_features_${feature_key}_seed${seed}_${timestamp}.json"

[[ -x "${python_bin}" ]] || { echo "[ERROR] missing ${python_bin}" >&2; exit 1; }
[[ -d "${data_root}" ]] || { echo "[ERROR] missing data root: ${data_root}" >&2; exit 1; }
[[ -d "${manifest_root}" ]] || { echo "[ERROR] missing clean manifest root: ${manifest_root}" >&2; exit 1; }
[[ -f "${manifest_root}/overlap_review_summary.json" ]] || { echo "[ERROR] missing completed overlap review summary" >&2; exit 1; }
[[ -f "${predictor_run_dir}/completed.json" ]] || { echo "[ERROR] predictor run is incomplete: ${predictor_run_dir}" >&2; exit 1; }
[[ -f "${config}" ]] || { echo "[ERROR] missing ${config}" >&2; exit 1; }
[[ -f "${checkpoint}" ]] || { echo "[ERROR] missing ${checkpoint}" >&2; exit 1; }
[[ ! -e "${output_dir}" ]] || { echo "[ERROR] output already exists: ${output_dir}" >&2; exit 1; }
tmux has-session -t "${session}" 2>/dev/null && { echo "[ERROR] tmux session exists: ${session}" >&2; exit 1; }

active_same_run="$(tmux list-sessions -F '#S' 2>/dev/null | grep -E "^s2ogsrfeat-${feature_key}-s${seed}-" || true)"
if [[ -n "${active_same_run}" ]]; then
  echo "[ERROR] the same OGSR feature export is already active:" >&2
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
echo "[PRELAUNCH] phase=OGSR-v0 feature export"
echo "[PRELAUNCH] feature_key=${feature_key}"
echo "[PRELAUNCH] sar_shuffle_mode=${sar_shuffle_mode}"
echo "[PRELAUNCH] audit=${audit_path}"
echo "[PRELAUNCH] predictor=${predictor_run_dir}"
echo "[PRELAUNCH] checkpoint=${checkpoint}"
echo "[PRELAUNCH] output=${output_dir}"
echo "[PRELAUNCH] log=${log_path}"

run_command="set -o pipefail; export CUDA_VISIBLE_DEVICES=${gpu}; cd \"${repo_root}\"; \"${python_bin}\" src/stage2/export_ogsr_v0_features.py --checkpoint \"${checkpoint}\" --config \"${config}\" --data-root \"${data_root}\" --output-dir \"${output_dir}\" --splits train val --seed \"${seed}\" --sar-shuffle-mode \"${sar_shuffle_mode}\" --sar-shuffle-seed 20260627 --batch-size 1 2>&1 | tee \"${log_path}\""
tmux new-session -d -s "${session}" bash -lc "${run_command}"

echo "[STARTED] session=${session}"
echo "[STARTED] audit=${audit_path}"
echo "[STARTED] log=${log_path}"
echo "[STARTED] output=${output_dir}"
echo "[MONITOR] tmux attach -t ${session}"
echo "[MONITOR] tail -f '${log_path}'"
echo "[STOP] tmux kill-session -t ${session}"
