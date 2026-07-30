#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: bash scripts/evaluate_stage2_v2_run.sh RUN_DIR GPU_INDEX [val|test] [grade|damage|damaged] [DAMAGE_MARGIN_THRESHOLD]" >&2
  exit 2
fi

run_dir="$(realpath "$1")"
gpu="$2"
split="${3:-test}"
checkpoint_kind="${4:-grade}"
damage_margin_threshold="${5:-}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GPU_INDEX must be an integer" >&2; exit 2; }
case "${split}" in val|test) ;; *) echo "[ERROR] split must be val or test" >&2; exit 2 ;; esac
case "${checkpoint_kind}" in
  grade) checkpoint_file="best_bo_grade_macro_f1.pth" ;;
  damage) checkpoint_file="best_bo_damage_macro_f1.pth" ;;
  damaged) checkpoint_file="best_bo_damaged_f1.pth" ;;
  *) echo "[ERROR] checkpoint must be grade, damage, or damaged" >&2; exit 2 ;;
esac
if [[ -n "${damage_margin_threshold}" && ! "${damage_margin_threshold}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ERROR] DAMAGE_MARGIN_THRESHOLD must be numeric, got ${damage_margin_threshold}" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
python_bin="/home/yr/miniconda3/envs/sam3/bin/python"
config="${run_dir}/config_resolved.json"
checkpoint="${run_dir}/checkpoints/${checkpoint_file}"
timestamp="$(date +%Y%m%d_%H%M%S)"
threshold_arg=""
threshold_suffix=""
if [[ -n "${damage_margin_threshold}" ]]; then
  threshold_label="${damage_margin_threshold//-/m}"
  threshold_label="${threshold_label//./p}"
  threshold_suffix="_thr_${threshold_label}"
  threshold_arg=" --damage-margin-threshold \"${damage_margin_threshold}\""
fi
output_dir="${run_dir}/${split}_best_${checkpoint_kind}${threshold_suffix}"
session="s2v2-eval-${split}-${checkpoint_kind}${threshold_suffix}-${timestamp}-${RANDOM}"
log_path="${run_dir}/eval_${split}_${checkpoint_kind}${threshold_suffix}_${timestamp}.log"

[[ -f "${run_dir}/completed.json" ]] || { echo "[ERROR] run is incomplete: ${run_dir}" >&2; exit 1; }
[[ -f "${config}" ]] || { echo "[ERROR] missing ${config}" >&2; exit 1; }
[[ -f "${checkpoint}" ]] || { echo "[ERROR] missing ${checkpoint}" >&2; exit 1; }
[[ ! -e "${output_dir}" ]] || { echo "[ERROR] evaluation output exists: ${output_dir}" >&2; exit 1; }

cd "${repo_root}"
echo "[PRELAUNCH] existing tmux sessions:"
tmux ls || true
nvidia-smi -i "${gpu}" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo "[PRELAUNCH] run=${run_dir}"
echo "[PRELAUNCH] checkpoint=${checkpoint}"
echo "[PRELAUNCH] split=${split}"
echo "[PRELAUNCH] damage_margin_threshold=${damage_margin_threshold:-argmax}"
echo "[PRELAUNCH] output=${output_dir}"

run_command="set -o pipefail; export CUDA_VISIBLE_DEVICES=${gpu}; cd \"${repo_root}\"; \"${python_bin}\" src/stage2/test_stage2_v2.py --config \"${config}\" --data-root \"${data_root}\" --checkpoint \"${checkpoint}\" --split \"${split}\" --output-dir \"${output_dir}\" --batch-size 4${threshold_arg} 2>&1 | tee \"${log_path}\""
tmux new-session -d -s "${session}" bash -lc "${run_command}"

echo "[STARTED] session=${session}"
echo "[STARTED] log=${log_path}"
echo "[STARTED] output=${output_dir}"
echo "[MONITOR] tmux attach -t ${session}"
echo "[STOP] tmux kill-session -t ${session}"
