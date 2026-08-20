#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <pilot|full> <unique-output-root> [candidate pilot-summary.json]" >&2
  exit 2
fi

PHASE=$1
OUTPUT_ROOT=$2
CANDIDATE=${3:-}
PILOT_SUMMARY=${4:-}

if [[ "$PHASE" != "pilot" && "$PHASE" != "full" ]]; then
  echo "phase must be pilot or full" >&2
  exit 2
fi
if [[ "$PHASE" == "full" && ( -z "$CANDIDATE" || -z "$PILOT_SUMMARY" ) ]]; then
  echo "full phase requires candidate and pilot summary" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse existing output root: $OUTPUT_ROOT" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE_ROOT=$(cd "$REPO_ROOT/.." && pwd)
RQ1_ROOT="$WORKSPACE_ROOT/outputs/stage2/rq1_paired_sar_nested_cv_v1_20260815"
MANIFEST_ROOT="$RQ1_ROOT/finalized_manifests"
DERANGEMENT_ROOT="$RQ1_ROOT/derangements"
SELECTION_ROOT="$RQ1_ROOT/stage2_nested_faststep_fullval_v3_20260817/selection"
DATA_ROOT="$WORKSPACE_ROOT/datasets/DisasterM3_optical_sar_damage_minimal_v0.2"

source /home/yr/miniconda3/etc/profile.d/conda.sh
conda activate sam3
cd "$REPO_ROOT"

bash instruction.sh stage2_v2_check_runtime

AVAILABLE_KB=$(df -Pk "$WORKSPACE_ROOT" | awk 'NR==2 {print $4}')
REQUIRED_KB=$((300 * 1024 * 1024))
if (( AVAILABLE_KB < REQUIRED_KB )); then
  echo "refusing formal RQ2 launch: less than 300 GiB free" >&2
  exit 2
fi

ACTIVE_GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$ACTIVE_GPU_PIDS" ]]; then
  echo "refusing formal RQ2 launch: GPU compute processes are already active" >&2
  echo "$ACTIVE_GPU_PIDS" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
SESSION="rq2_${PHASE}_${STAMP}"
LOG_ROOT="$OUTPUT_ROOT/launcher_logs"
LOG_PATH="$LOG_ROOT/${SESSION}.log"
mkdir -p "$LOG_ROOT"

ARGS=(
  python scripts/run_rq2_model_development.py
  --phase "$PHASE"
  --manifest-root "$MANIFEST_ROOT"
  --derangement-root "$DERANGEMENT_ROOT"
  --data-root "$DATA_ROOT"
  --selection-root "$SELECTION_ROOT"
  --output-root "$OUTPUT_ROOT"
  --gpus 0 1
)
if [[ "$PHASE" == "full" ]]; then
  ARGS+=(--candidate "$CANDIDATE" --pilot-summary "$PILOT_SUMMARY")
fi

printf -v COMMAND '%q ' "${ARGS[@]}"
tmux new-session -d -s "$SESSION" \
  "bash -lc 'source /home/yr/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && cd \"$REPO_ROOT\" && $COMMAND 2>&1 | tee \"$LOG_PATH\"'"

echo "session=$SESSION"
echo "log=$LOG_PATH"
echo "output=$OUTPUT_ROOT"
echo "monitor=tmux attach -t $SESSION"
echo "stop=tmux send-keys -t $SESSION C-c"
