#!/usr/bin/env bash
# Run RepOpt VGG end-to-end classification (NPU L0+L1 + CPU 512-feature classifier)
# Usage: ./run_vgg_e2e.sh [image_index]
#   image_index: 0-9999 CIFAR-10 test sample (default: 0)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMG_IDX="${1:-0}"
OUT_DIR="$ROOT/sim/vgg_e2e"

echo "=== Clean ==="
rm -rf "$OUT_DIR"

echo "=== Generate (img_idx=$IMG_IDX) ==="
python3 -B "$ROOT/tools/pth/gen_vgg_e2e.py" --out-dir "$OUT_DIR" --img-idx "$IMG_IDX"

echo "=== Compile ==="
verilator --binary --timing \
  -I"$OUT_DIR" -Mdir "$OUT_DIR/obj_dir" --top-module tb_soc_vgg_e2e \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSED -Wno-UNDRIVEN \
  -Wno-UNOPTFLAT -Wno-PINMISSING -Wno-CASEINCOMPLETE -Wno-COMBDLY \
  -Wno-INITIALDLY -Wno-LITENDIAN \
  "$ROOT/sim/picorv32.v" \
  "$ROOT/rtl/pe/"*.v \
  "$ROOT/rtl/common/"*.v \
  "$ROOT/rtl/buf/"*.v \
  "$ROOT/rtl/array/"*.v \
  "$ROOT/rtl/axi/"*.v \
  "$ROOT/rtl/ctrl/"*.v \
  "$ROOT/rtl/power/"*.v \
  "$ROOT/rtl/soc/"*.v \
  "$ROOT/rtl/top/"*.v \
  "$ROOT/tb/tb_soc_vgg_e2e.v"

echo "=== Run ==="
timeout 120 "$OUT_DIR/obj_dir/Vtb_soc_vgg_e2e"
