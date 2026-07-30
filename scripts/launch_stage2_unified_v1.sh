#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/launch_stage2_unified_v1.sh {paired|shuffled} {42|3407|2026} GPU_INDEX" >&2
  exit 2
fi

variant="${1,,}"
seed="$2"
gpu="$3"
case "${variant}" in
  paired)
    experiment="S2U1_UABCD_paired"
    config="configs/stage2_unified_v1/uabcd_paired.yaml"
    ;;
  shuffled)
    experiment="S2U1_UABCD_shuffled"
    config="configs/stage2_unified_v1/uabcd_shuffled.yaml"
    ;;
  *) echo "[ERROR] unknown unified-v1 variant: ${variant}" >&2; exit 2 ;;
esac
case "${seed}" in
  42|3407|2026) ;;
  *) echo "[ERROR] formal seed must be 42, 3407, or 2026" >&2; exit 2 ;;
esac
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GPU_INDEX must be an integer" >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
manifest_root="${repo_root}/manifests/stage2_v2_clean_human_reviewed_20260624"
python_bin="/home/yr/miniconda3/envs/sam3/bin/python"
timestamp="$(date +%Y%m%d_%H%M%S)"
session="s2u1-${variant}-s${seed}-${timestamp}"
run_dir="${repo_root}/outputs/stage2/unified_v1/${experiment}/seed_${seed}/run_${timestamp}"
log_path="${repo_root}/logs/stage2_unified_v1_${variant}_seed${seed}_${timestamp}.log"
audit_path="${repo_root}/outputs/stage2/unified_v1_preflight/audit_${variant}_seed${seed}_${timestamp}.json"
protocol_path="${repo_root}/outputs/stage2/unified_v1_preflight/protocol_${variant}_seed${seed}_${timestamp}.json"
gate_path="${repo_root}/outputs/stage2/unified_v1/gates/seed42_gate.json"

[[ -x "${python_bin}" ]] || { echo "[ERROR] missing ${python_bin}" >&2; exit 1; }
[[ -d "${data_root}" ]] || { echo "[ERROR] missing data root: ${data_root}" >&2; exit 1; }
[[ -d "${manifest_root}" ]] || { echo "[ERROR] missing manifest root: ${manifest_root}" >&2; exit 1; }
[[ -f "${repo_root}/${config}" ]] || { echo "[ERROR] missing config: ${config}" >&2; exit 1; }
[[ ! -e "${run_dir}" ]] || { echo "[ERROR] output already exists: ${run_dir}" >&2; exit 1; }

if [[ "${seed}" != "42" ]]; then
  [[ -f "${gate_path}" ]] || {
    echo "[ERROR] three-seed expansion is locked until seed-42 paired/shuffled gate passes." >&2
    exit 1
  }
  "${python_bin}" - "${gate_path}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("status") != "pass":
    raise SystemExit("[ERROR] seed-42 advancement gate did not pass")
PY
fi

tmux has-session -t "${session}" 2>/dev/null && {
  echo "[ERROR] tmux session exists: ${session}" >&2
  exit 1
}
active_same_run="$(tmux list-sessions -F '#S' 2>/dev/null | grep -E "^s2u1-${variant}-s${seed}-" || true)"
if [[ -n "${active_same_run}" ]]; then
  echo "[ERROR] the same unified-v1 variant/seed is already active:" >&2
  echo "${active_same_run}" >&2
  exit 1
fi

gpu_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$' || true)"
if [[ -n "${gpu_pids}" && "${STAGE2_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "[ERROR] GPU ${gpu} already has compute processes: ${gpu_pids//$'\n'/, }" >&2
  exit 1
fi

cd "${repo_root}"
"${python_bin}" scripts/validate_stage2_unified_protocol.py --out "${protocol_path}"
env \
  PATH=/home/yr/miniconda3/envs/sam3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  STAGE2_DATA_ROOT="${data_root}" \
  STAGE2_V2_STRICT_MANIFEST_DIR="${manifest_root}" \
  STAGE2_V2_STRICT_AUDIT_OUT="${audit_path}" \
  bash instruction.sh stage2_v2_strict_check_runtime

echo "[PRELAUNCH] existing tmux sessions:"
tmux ls || true
echo "[PRELAUNCH] selected GPU:"
nvidia-smi -i "${gpu}" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo "[PRELAUNCH] protocol=stage2_unified_v1"
echo "[PRELAUNCH] selection_split=val (test embargoed)"
echo "[PRELAUNCH] variant=${variant}"
echo "[PRELAUNCH] seed=${seed}"
echo "[PRELAUNCH] audit=${audit_path}"
echo "[PRELAUNCH] protocol_validation=${protocol_path}"
echo "[PRELAUNCH] config=${repo_root}/${config}"
echo "[PRELAUNCH] output=${run_dir}"
echo "[PRELAUNCH] log=${log_path}"

run_command="set -o pipefail; export CUDA_VISIBLE_DEVICES=${gpu}; export STAGE2_DATA_AUDIT_PATH=\"${audit_path}\"; cd \"${repo_root}\"; \"${python_bin}\" src/stage2/train_stage2_v2.py --config \"${config}\" --data-root \"${data_root}\" --output-dir \"${run_dir}\" --seed \"${seed}\" 2>&1 | tee \"${log_path}\""
tmux new-session -d -s "${session}" bash -lc "${run_command}"

echo "[STARTED] session=${session}"
echo "[STARTED] log=${log_path}"
echo "[STARTED] output=${run_dir}"
echo "[MONITOR] tmux attach -t ${session}"
echo "[MONITOR] tail -f '${log_path}'"
echo "[STOP] tmux kill-session -t ${session}"
