#!/usr/bin/env bash
set -euo pipefail

# Stage-1 formal training launcher.
#
# This script is intentionally argument-driven:
#   bash train.sh o1      # train + test U-Net ResNet34 frequent-disaster baseline
#   bash train.sh o2      # train + test U-Net ResNet34 all-disaster comparison
#   bash train.sh o3      # train + test DeepLabV3+ ResNet50 frequent baseline
#   bash train.sh o4      # train + test SegFormer-B0 frequent baseline
#   bash train.sh all     # run O1 -> O2 -> O3 -> O4 in order
#
# It does not start training when run without an argument.
# For fast exploration, enable early stopping via environment variables, e.g.:
#   EARLY_STOPPING_PATIENCE=20 EARLY_STOPPING_MIN_EPOCHS=30 bash train.sh o4

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-${SCRIPT_DIR}}"
cd "${CODE_ROOT}"
mkdir -p logs

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_step() {
  local name="$1"
  shift
  echo "[$(timestamp)] START ${name}"
  "$@" 2>&1 | tee "logs/${name}.log"
  echo "[$(timestamp)] DONE ${name}"
}

run_o1() {
  # O1: main baseline.
  # Train on frequent disasters only: earthquake, hurricane, volcano.
  # Evaluate on all / frequent / rare / ok-only test splits via instruction.sh.
  run_step "O1_unet_resnet34_freq.train" bash instruction.sh train_unet_freq
  run_step "O1_unet_resnet34_freq.test" bash instruction.sh test_unet_freq
}

run_o2() {
  # O2: data-filter ablation.
  # Same U-Net model, but train on all QC-clean SAR-aligned Stage-1 samples.
  # Compare this against O1 to quantify whether dropping rare disasters helps.
  run_step "O2_unet_resnet34_all.train" bash instruction.sh train_unet_all
  run_step "O2_unet_resnet34_all.test" bash instruction.sh test_unet_all
}

run_o3() {
  # O3: stronger CNN baseline.
  # DeepLabV3+ ResNet50, trained on frequent-disaster core set.
  run_step "O3_deeplabv3p_resnet50_freq.train" bash instruction.sh train_deeplab_freq
  run_step "O3_deeplabv3p_resnet50_freq.test" bash instruction.sh test_deeplab_freq
}

run_o4() {
  # O4: lightweight Transformer segmentation baseline.
  # SegFormer-B0, trained on frequent-disaster core set.
  # Pretrained mit_b0 weights must already exist in the torch hub cache:
  # /home/yr/.cache/torch/hub/checkpoints/mit_b0.pth
  run_step "O4_segformer_b0_freq.train" bash instruction.sh train_segformer_freq
  run_step "O4_segformer_b0_freq.test" bash instruction.sh test_segformer_freq
}

summarize_results() {
  # Aggregate metrics.json files under outputs/*/test*/ into outputs/stage1_summary.csv.
  run_step "stage1_summary" bash instruction.sh summarize
}

usage() {
  cat <<'EOF'
Usage: bash train.sh COMMAND

Commands:
  o1         Train/test U-Net ResNet34 frequent-disaster baseline
  o2         Train/test U-Net ResNet34 all-disaster comparison
  o3         Train/test DeepLabV3+ ResNet50 frequent baseline
  o4         Train/test SegFormer-B0 frequent baseline
  all        Run O1, O2, O3, O4, then summarize
  summarize  Summarize completed test metrics

Examples:
  CUDA_VISIBLE_DEVICES=0 bash train.sh o1
  CUDA_VISIBLE_DEVICES=0 bash train.sh all
  CUDA_VISIBLE_DEVICES=0 EARLY_STOPPING_PATIENCE=20 EARLY_STOPPING_MIN_EPOCHS=30 bash train.sh o4
EOF
}

case "${1:-}" in
  o1) run_o1 ;;
  o2) run_o2 ;;
  o3) run_o3 ;;
  o4) run_o4 ;;
  all)
    run_o1
    run_o2
    run_o3
    run_o4
    summarize_results
    ;;
  summarize) summarize_results ;;
  *) usage; exit 1 ;;
esac
