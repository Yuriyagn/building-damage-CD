#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/launch_stage2_metadata_multitask_queue.sh GPU_INDEX [m0_m1|m2]" >&2
  exit 2
fi

gpu="$1"
phase="${2:-m0_m1}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GPU_INDEX must be an integer" >&2; exit 2; }
case "${phase}" in m0_m1|m2) ;; *) echo "[ERROR] phase must be m0_m1 or m2" >&2; exit 2 ;; esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
manifest_root="${repo_root}/manifests/stage2_v2_event_group_bright_r4_predicted_v1_20260803"
audit_path="${repo_root}/outputs/stage2/metadata_multitask_v1/preflight/train_val_data_audit.json"
python_bin="${PYTHON_BIN:-$(command -v python)}"
timestamp="$(date +%Y%m%d_%H%M%S)"
session="s2meta-${phase//_/-}-${timestamp}"
queue_log="${repo_root}/logs/stage2_metadata_queue_${phase}_${timestamp}.log"

[[ -x "${python_bin}" ]] || { echo "[ERROR] missing Python: ${python_bin}" >&2; exit 1; }
[[ -d "${data_root}" ]] || { echo "[ERROR] missing data root: ${data_root}" >&2; exit 1; }
[[ -f "${manifest_root}/predicted_prior/train.jsonl" ]] || { echo "[ERROR] missing R4 train manifest" >&2; exit 1; }
[[ -f "${manifest_root}/predicted_prior/val.jsonl" ]] || { echo "[ERROR] missing R4 val manifest" >&2; exit 1; }

if [[ "${phase}" == "m2" && ! -f "${repo_root}/configs/stage2_metadata_multitask_v1/shuffled_labels_seed20260814.json" ]]; then
  echo "[ERROR] M2 label map has not been generated" >&2
  exit 1
fi

gpu_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$' || true)"
if [[ -n "${gpu_pids}" ]]; then
  echo "[ERROR] GPU ${gpu} already has compute processes: ${gpu_pids//$'\n'/, }" >&2
  exit 1
fi
tmux has-session -t "${session}" 2>/dev/null && { echo "[ERROR] tmux session exists: ${session}" >&2; exit 1; }

cd "${repo_root}"
if [[ ! -f "${audit_path}" ]]; then
  "${python_bin}" scripts/audit_stage2_metadata_train_val.py \
    --data-root "${data_root}" \
    --train-manifest "${manifest_root}/predicted_prior/train.jsonl" \
    --val-manifest "${manifest_root}/predicted_prior/val.jsonl" \
    --out "${audit_path}"
fi
STAGE2_DATA_ROOT="${data_root}" STAGE2_MANIFEST_DIR="${manifest_root}" \
  bash instruction.sh stage2_v2_check_runtime
"${python_bin}" - "${audit_path}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("splits_read") != ["train", "val"] or payload.get("test_read") is not False:
    raise SystemExit("metadata audit violates the train/validation-only protocol")
if payload.get("hard_error_count") != 0:
    raise SystemExit("metadata train/validation audit contains hard errors")
if payload.get("cross_split_full_sample_duplicate_group_count") != 0:
    raise SystemExit("metadata train/validation audit contains cross-split duplicates")
print("[OK] train/validation-only full-content audit is eligible for the comparison")
PY

run_command="set -o pipefail; export STAGE2_DATA_ROOT=\"${data_root}\"; export PYTHON_BIN=\"${python_bin}\"; cd \"${repo_root}\"; bash scripts/run_stage2_metadata_multitask_queue.sh \"${gpu}\" \"${phase}\" 2>&1 | tee \"${queue_log}\""
tmux new-session -d -s "${session}" bash -lc "${run_command}"

echo "[STARTED] session=${session}"
echo "[STARTED] phase=${phase}"
echo "[STARTED] queue_log=${queue_log}"
echo "[STARTED] output_root=${repo_root}/outputs/stage2/metadata_multitask_v1"
echo "[MONITOR] tmux attach -t ${session}"
echo "[STOP] tmux kill-session -t ${session}"
