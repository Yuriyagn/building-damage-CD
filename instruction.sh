#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CODE_ROOT="${CODE_ROOT:-${SCRIPT_DIR}}"
export DATA_ROOT="${DATA_ROOT:-/home/yr/code/datasets/DisasterM3_stage1_optical_building}"
export OUT_ROOT="${OUT_ROOT:-${CODE_ROOT}/outputs}"

cd "${CODE_ROOT}"
mkdir -p "${OUT_ROOT}" logs

show_paths() {
  echo "[INFO] CODE_ROOT=${CODE_ROOT}"
  echo "[INFO] DATA_ROOT=${DATA_ROOT}"
  echo "[INFO] OUT_ROOT=${OUT_ROOT}"
}

check_env() {
  show_paths
  python - <<'PY'
import importlib.util
import sys
import torch

print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
if importlib.util.find_spec("segmentation_models_pytorch") is None:
    raise SystemExit("segmentation_models_pytorch is not installed")
import segmentation_models_pytorch as smp
print("smp:", getattr(smp, "__version__", "installed"))
PY
}

check_data() {
  show_paths
  python tools/check_stage1_dataset.py \
    --data-root "${DATA_ROOT}" \
    --manifest "${DATA_ROOT}/manifests/building_extract_train_sar_qc_freq.jsonl" \
    --check-mask-values
  python tools/check_stage1_dataset.py \
    --data-root "${DATA_ROOT}" \
    --manifest "${DATA_ROOT}/manifests/building_extract_val_sar_qc.jsonl"
  python tools/check_stage1_dataset.py \
    --data-root "${DATA_ROOT}" \
    --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc.jsonl"
}

smoke_unet_freq() {
  python src/train.py \
    --config configs/stage1_optical_building_unet_resnet34_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/debug_smoke_unet_freq" \
    --smoke-test
}

overfit_unet_freq() {
  python src/train.py \
    --config configs/stage1_optical_building_unet_resnet34_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/debug_overfit_unet_freq" \
    --overfit-batches 2 \
    --epochs 30
}

train_unet_freq() {
  python src/train.py \
    --config configs/stage1_optical_building_unet_resnet34_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/O1_unet_resnet34_freq"
}

test_unet_freq() {
  local ckpt="${OUT_ROOT}/O1_unet_resnet34_freq/checkpoints/best_iou.pth"
  python src/test.py --config configs/stage1_optical_building_unet_resnet34_freq.yaml --data-root "${DATA_ROOT}" --checkpoint "${ckpt}" --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc.jsonl" --output-dir "${OUT_ROOT}/O1_unet_resnet34_freq/test_all"
  python src/test.py --config configs/stage1_optical_building_unet_resnet34_freq.yaml --data-root "${DATA_ROOT}" --checkpoint "${ckpt}" --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc_freq.jsonl" --output-dir "${OUT_ROOT}/O1_unet_resnet34_freq/test_freq"
  python src/test.py --config configs/stage1_optical_building_unet_resnet34_freq.yaml --data-root "${DATA_ROOT}" --checkpoint "${ckpt}" --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc_rare.jsonl" --output-dir "${OUT_ROOT}/O1_unet_resnet34_freq/test_rare"
  python src/test.py --config configs/stage1_optical_building_unet_resnet34_freq.yaml --data-root "${DATA_ROOT}" --checkpoint "${ckpt}" --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc_ok.jsonl" --output-dir "${OUT_ROOT}/O1_unet_resnet34_freq/test_ok"
}

train_unet_all() {
  python src/train.py \
    --config configs/stage1_optical_building_unet_resnet34_all.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/O2_unet_resnet34_all"
}

test_unet_all() {
  python src/test.py \
    --config configs/stage1_optical_building_unet_resnet34_all.yaml \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${OUT_ROOT}/O2_unet_resnet34_all/checkpoints/best_iou.pth" \
    --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc.jsonl" \
    --output-dir "${OUT_ROOT}/O2_unet_resnet34_all/test_all"
}

train_deeplab_freq() {
  python src/train.py \
    --config configs/stage1_optical_building_deeplabv3p_resnet50_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/O3_deeplabv3p_resnet50_freq"
}

test_deeplab_freq() {
  python src/test.py \
    --config configs/stage1_optical_building_deeplabv3p_resnet50_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${OUT_ROOT}/O3_deeplabv3p_resnet50_freq/checkpoints/best_iou.pth" \
    --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc.jsonl" \
    --output-dir "${OUT_ROOT}/O3_deeplabv3p_resnet50_freq/test_all"
}

train_segformer_freq() {
  python src/train.py \
    --config configs/stage1_optical_building_segformer_b0_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/O4_segformer_b0_freq"
}

test_segformer_freq() {
  python src/test.py \
    --config configs/stage1_optical_building_segformer_b0_freq.yaml \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${OUT_ROOT}/O4_segformer_b0_freq/checkpoints/best_iou.pth" \
    --manifest "${DATA_ROOT}/manifests/building_extract_test_sar_qc.jsonl" \
    --output-dir "${OUT_ROOT}/O4_segformer_b0_freq/test_all"
}

summarize() {
  python tools/summarize_stage1_results.py \
    --outputs-root "${OUT_ROOT}" \
    --out "${OUT_ROOT}/stage1_summary.csv"
}

case "${1:-}" in
  check_env) check_env ;;
  check_data) check_data ;;
  smoke_unet_freq) smoke_unet_freq ;;
  overfit_unet_freq) overfit_unet_freq ;;
  train_unet_freq) train_unet_freq ;;
  test_unet_freq) test_unet_freq ;;
  train_unet_all) train_unet_all ;;
  test_unet_all) test_unet_all ;;
  train_deeplab_freq) train_deeplab_freq ;;
  test_deeplab_freq) test_deeplab_freq ;;
  train_segformer_freq) train_segformer_freq ;;
  test_segformer_freq) test_segformer_freq ;;
  summarize) summarize ;;
  *)
    echo "Usage: bash instruction.sh {check_env|check_data|smoke_unet_freq|overfit_unet_freq|train_unet_freq|test_unet_freq|train_unet_all|test_unet_all|train_deeplab_freq|test_deeplab_freq|train_segformer_freq|test_segformer_freq|summarize}"
    exit 1
    ;;
esac
