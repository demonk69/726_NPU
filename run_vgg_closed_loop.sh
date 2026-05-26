#!/usr/bin/env bash
# Run runtime closed-loop RepOpt VGG classification.
#
# Usage:
#   ./run_vgg_closed_loop.sh
#   ./run_vgg_closed_loop.sh 7
#   ./run_vgg_closed_loop.sh --image cat.jpg
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$ROOT/sim/vgg_closed_loop"
TIMEOUT_CYCLES="${VGG_CLOSED_TIMEOUT_CYCLES:-150000000}"

GEN_ARGS=()
GEN_ARGS+=(--timeout-cycles "$TIMEOUT_CYCLES")
if [[ "${1:-}" == "--image" ]]; then
    GEN_ARGS+=(--image "${2:?missing image path}")
else
    GEN_ARGS+=(--img-idx "${1:-0}")
fi

CLASSES=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck")

echo "=== Clean ==="
rm -rf "$OUT_DIR"

echo "=== Generate Closed Loop ==="
python3 -B "$ROOT/tools/pth/gen_vgg_closed_loop.py" --out-dir "$OUT_DIR" "${GEN_ARGS[@]}"

echo "=== Compile ==="
verilator --binary --timing \
  -I"$OUT_DIR" -Mdir "$OUT_DIR/obj_dir" --top-module tb_soc_vgg_closed_loop \
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
  "$ROOT/tb/tb_soc_vgg_closed_loop.v"

echo "=== Run ==="
LOG_FILE="$OUT_DIR/run.log"
set +e
timeout 2400 stdbuf -oL -eL "$OUT_DIR/obj_dir/Vtb_soc_vgg_closed_loop" 2>&1 | tee "$LOG_FILE"
RUN_RC=${PIPESTATUS[0]}
set -e

grep -E '\[PASS\]|\[FAIL\]|\[TIMEOUT\]|Cycles' "$LOG_FILE" || true

if grep -qE '\[FAIL\]|\[TIMEOUT\]' "$LOG_FILE"; then
    RUN_RC=1
elif ! grep -q '\[PASS\]' "$LOG_FILE"; then
    RUN_RC=1
fi

PRED=$(grep -oP 'Predicted class: \K\d+' "$LOG_FILE" || true)
if [[ -n "$PRED" ]]; then
    echo "  Predicted: ${CLASSES[$PRED]} (class $PRED)"
fi

exit "$RUN_RC"
