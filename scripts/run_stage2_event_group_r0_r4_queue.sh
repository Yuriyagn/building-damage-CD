#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_stage2_event_group_r0_r4_queue.sh {r0|r4} GPU_INDEX" >&2
  exit 2
fi

experiment_key="${1,,}"
gpu="$2"
case "${experiment_key}" in r0|r4) ;; *) echo "[ERROR] comparison must be r0 or r4" >&2; exit 2 ;; esac
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GPU_INDEX must be an integer" >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

for seed in 42 3407 2026; do
  echo "[QUEUE] launching comparison=${experiment_key} seed=${seed} gpu=${gpu}"
  launch_output="$(bash scripts/launch_stage2_event_group_r0_r4.sh "${experiment_key}" "${seed}" "${gpu}" 2>&1)"
  echo "${launch_output}"
  session="$(sed -n 's/^\[STARTED\] session=//p' <<<"${launch_output}" | tail -n 1)"
  run_dir="$(sed -n 's/^\[STARTED\] output=//p' <<<"${launch_output}" | tail -n 1)"
  log_path="$(sed -n 's/^\[STARTED\] log=//p' <<<"${launch_output}" | tail -n 1)"
  if [[ -z "${session}" || -z "${run_dir}" || -z "${log_path}" ]]; then
    echo "[ERROR] launcher did not return complete run metadata" >&2
    exit 1
  fi
  while tmux has-session -t "${session}" 2>/dev/null; do
    sleep 30
  done
  if [[ ! -f "${run_dir}/completed.json" ]]; then
    echo "[ERROR] run ended without completed.json: ${run_dir}" >&2
    tail -n 80 "${log_path}" >&2 || true
    exit 1
  fi
  echo "[QUEUE] completed comparison=${experiment_key} seed=${seed} output=${run_dir}"
done

echo "[QUEUE] all validation-training runs completed for ${experiment_key}; test remains embargoed"
