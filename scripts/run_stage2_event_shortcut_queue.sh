#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <cuda_device> <d1|d2|e1|d4:event> [...]" >&2
  exit 2
fi

cuda_device="$1"
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
python_bin="${PYTHON_BIN:-$(command -v python)}"
data_root="${STAGE2_DATA_ROOT:-${workspace_root}/datasets/DisasterM3_optical_sar_damage_minimal_v0.2}"
base_audit="$repo_root/outputs/stage2/event_shortcut_v1_preflight/base_strict_audit_20260803.json"

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$cuda_device"

for selector in "$@"; do
  case "$selector" in
    d1)
      experiment="D1_pre_only"
      config="configs/stage2_event_shortcut_v1/d1_pre_only.yaml"
      audit="$base_audit"
      ;;
    d2)
      experiment="D2_background_only"
      config="configs/stage2_event_shortcut_v1/d2_background_only.yaml"
      audit="$base_audit"
      ;;
    e1)
      experiment="E1_event_class_balanced"
      config="configs/stage2_event_shortcut_v1/e1_event_class_balanced.yaml"
      audit="$base_audit"
      ;;
    d4:*)
      event="${selector#d4:}"
      experiment="D4_LOEO_${event}"
      config="configs/stage2_event_shortcut_v1/d4_loeo/${event}.yaml"
      audit="$repo_root/manifests/stage2_event_shortcut_v1/d4_loeo/${event}/audit.json"
      ;;
    *)
      echo "unknown selector: $selector" >&2
      exit 2
      ;;
  esac

  [[ -x "$python_bin" ]] || { echo "missing Python: $python_bin" >&2; exit 1; }
  [[ -d "$data_root" ]] || { echo "missing data root: $data_root" >&2; exit 1; }
  [[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 1; }
  [[ -f "$audit" ]] || { echo "missing audit: $audit" >&2; exit 1; }

  stamp="$(date +%Y%m%d_%H%M%S)"
  run_dir="outputs/stage2/event_shortcut_v1/${experiment}/seed_42/run_${stamp}"
  log_path="logs/stage2_event_shortcut_${experiment}_seed42_${stamp}.log"
  [[ ! -e "$run_dir" ]] || { echo "refusing existing output: $run_dir" >&2; exit 1; }

  export STAGE2_DATA_AUDIT_PATH="$audit"
  echo "[$(date --iso-8601=seconds)] START selector=$selector gpu=$cuda_device run=$run_dir"
  "$python_bin" src/stage2/train_stage2_v2.py \
    --config "$config" \
    --data-root "$data_root" \
    --output-dir "$run_dir" \
    --seed 42 2>&1 | tee "$log_path"

  checkpoint="$run_dir/checkpoints/best_bo_grade_macro_f1.pth"
  [[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
  "$python_bin" src/stage2/test_stage2_v2.py \
    --config "$config" \
    --data-root "$data_root" \
    --checkpoint "$checkpoint" \
    --split val \
    --output-dir "$run_dir/val_best_grade" \
    --batch-size 4 \
    --num-workers 4 \
    --max-previews 0 2>&1 | tee -a "$log_path"
  echo "[$(date --iso-8601=seconds)] DONE selector=$selector gpu=$cuda_device run=$run_dir"
done
