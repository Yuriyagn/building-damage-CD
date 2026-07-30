#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/launch_stage2_v2_phase4_binary_aux.sh {f1|f2|f3} GPU_INDEX [SEED]" >&2
  exit 2
fi

experiment_key="${1,,}"
gpu="$2"
seed="${3:-42}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GPU_INDEX must be an integer" >&2; exit 2; }
[[ "${seed}" =~ ^[0-9]+$ ]] || { echo "[ERROR] SEED must be an integer" >&2; exit 2; }

case "${experiment_key}" in
  f1)
    experiment="V2_BINAUX_F1_p2_w0p1"
    config="configs/stage2_v2_phase4_binary_aux/v2_binaux_f1_p2_weight_0p1.yaml"
    diagnostic="P2 binary auxiliary loss, binary_aux_weight=0.1, binary_pos_weight=1.0"
    ;;
  f2)
    experiment="V2_BINAUX_F2_p2_w0p5"
    config="configs/stage2_v2_phase4_binary_aux/v2_binaux_f2_p2_weight_0p5.yaml"
    diagnostic="P2 binary auxiliary loss, binary_aux_weight=0.5, binary_pos_weight=1.0"
    ;;
  f3)
    experiment="V2_BINAUX_F3_p2_w0p3_posw2"
    config="configs/stage2_v2_phase4_binary_aux/v2_binaux_f3_p2_weight_0p3_posw2.yaml"
    diagnostic="P2 binary auxiliary loss, binary_aux_weight=0.3, binary_pos_weight=2.0"
    ;;
  *) echo "[ERROR] unknown Phase 4 binary auxiliary experiment: ${experiment_key}" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
manifest_root="${repo_root}/manifests/stage2_v2_clean_human_reviewed_20260624"
python_bin="/home/yr/miniconda3/envs/sam3/bin/python"
timestamp="$(date +%Y%m%d_%H%M%S)"
session="s2v2bin-${experiment_key}-s${seed}-${timestamp}"
run_dir="${repo_root}/outputs/stage2/v2_phase4_binary_aux/${experiment}/seed_${seed}/run_${timestamp}"
log_path="${repo_root}/logs/stage2_v2_phase4_binary_aux_${experiment_key}_seed${seed}_${timestamp}.log"
audit_path="${repo_root}/outputs/stage2/v2_phase4_binary_aux_preflight/audit_${experiment_key}_seed${seed}_${timestamp}.json"

[[ -x "${python_bin}" ]] || { echo "[ERROR] missing ${python_bin}" >&2; exit 1; }
[[ -d "${data_root}" ]] || { echo "[ERROR] missing data root: ${data_root}" >&2; exit 1; }
[[ -d "${manifest_root}" ]] || { echo "[ERROR] missing clean manifest root: ${manifest_root}" >&2; exit 1; }
[[ -f "${manifest_root}/overlap_review_summary.json" ]] || { echo "[ERROR] missing completed overlap review summary" >&2; exit 1; }
[[ -f "${repo_root}/${config}" ]] || { echo "[ERROR] missing config: ${config}" >&2; exit 1; }
[[ ! -e "${run_dir}" ]] || { echo "[ERROR] output already exists: ${run_dir}" >&2; exit 1; }
tmux has-session -t "${session}" 2>/dev/null && { echo "[ERROR] tmux session exists: ${session}" >&2; exit 1; }

active_same_run="$(tmux list-sessions -F '#S' 2>/dev/null | grep -E "^s2v2bin-${experiment_key}-s${seed}-" || true)"
if [[ -n "${active_same_run}" ]]; then
  echo "[ERROR] the same Phase 4 binary auxiliary experiment/seed is already active:" >&2
  echo "${active_same_run}" >&2
  exit 1
fi

gpu_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$' || true)"
if [[ -n "${gpu_pids}" && "${STAGE2_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "[ERROR] GPU ${gpu} already has compute processes: ${gpu_pids//$'\n'/, }" >&2
  echo "[ERROR] launch one Phase 4 process per GPU, or set STAGE2_ALLOW_BUSY_GPU=1 deliberately." >&2
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
echo "[PRELAUNCH] phase=binary auxiliary loss tuning"
echo "[PRELAUNCH] diagnostic=${diagnostic}"
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
