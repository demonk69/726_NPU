#!/usr/bin/env bash
# Run runtime closed-loop RepOpt VGG classification.
#
# Usage:
#   ./run_vgg_closed_loop.sh
#   ./run_vgg_closed_loop.sh 7
#   ./run_vgg_closed_loop.sh --image cat.jpg
#   ./run_vgg_closed_loop.sh --shape 8x8 --image cat.jpg
#   ./run_vgg_closed_loop.sh --flow ws --shape 4x4 --image cat.jpg
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$ROOT/sim/vgg_closed_loop"
TIMEOUT_CYCLES="${VGG_CLOSED_TIMEOUT_CYCLES:-150000000}"

IMG_IDX="0"
IMAGE=""
SHAPE="16x16"
FLOW="os"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            IMAGE="${2:?missing image path}"
            shift 2
            ;;
        --shape)
            SHAPE="${2:?missing shape}"
            shift 2
            ;;
        --flow)
            FLOW="${2:?missing flow}"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [img_idx] [--image <file>] [--shape 4x4|8x8|16x16|8x32] [--flow os|ws]"
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            IMG_IDX="$1"
            shift
            ;;
    esac
done

case "$SHAPE" in
    4x4|8x8|16x16|8x32) ;;
    *)
        echo "Invalid shape: $SHAPE (expected 4x4, 8x8, 16x16, or 8x32)" >&2
        exit 2
        ;;
esac

case "$FLOW" in
    os|ws) ;;
    *)
        echo "Invalid flow: $FLOW (expected os or ws)" >&2
        exit 2
        ;;
esac

GEN_ARGS=()
GEN_ARGS+=(--timeout-cycles "$TIMEOUT_CYCLES")
GEN_ARGS+=(--shape "$SHAPE")
GEN_ARGS+=(--flow "$FLOW")
if [[ -n "$IMAGE" ]]; then
    GEN_ARGS+=(--image "$IMAGE")
else
    GEN_ARGS+=(--img-idx "$IMG_IDX")
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
