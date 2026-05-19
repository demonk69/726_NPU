#!/usr/bin/env bash
# Run RepOpt VGG end-to-end classification (NPU L0+L1 + CPU 512-feature classifier)
#
# Usage:
#   ./run_vgg_e2e.sh                    # CIFAR-10 image index 0
#   ./run_vgg_e2e.sh 7                  # CIFAR-10 image index 7
#   ./run_vgg_e2e.sh --image cat.jpg    # arbitrary image (resized to 32x32)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$ROOT/sim/vgg_e2e"

GEN_ARGS=()
if [[ "${1:-}" == "--image" ]]; then
    GEN_ARGS+=(--image "${2:?missing image path}")
else
    GEN_ARGS+=(--img-idx "${1:-0}")
fi

echo "=== Clean ==="
rm -rf "$OUT_DIR"

echo "=== Generate ==="
python3 -B "$ROOT/tools/pth/gen_vgg_e2e.py" --out-dir "$OUT_DIR" "${GEN_ARGS[@]}"

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
