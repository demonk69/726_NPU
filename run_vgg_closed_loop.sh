#!/usr/bin/env bash
# Run runtime closed-loop RepOpt VGG classification.
#
# Usage:
#   ./run_vgg_closed_loop.sh
#   ./run_vgg_closed_loop.sh 7
#   ./run_vgg_closed_loop.sh --image cat.jpg
#   ./run_vgg_closed_loop.sh --shape 8x8 --image cat.jpg
#   ./run_vgg_closed_loop.sh --flow ws --shape 4x4 --image cat.jpg
#   ./run_vgg_closed_loop.sh --shape 16x16 --flow os --lanes 2
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$ROOT/sim/vgg_closed_loop"
TIMEOUT_CYCLES="${VGG_CLOSED_TIMEOUT_CYCLES:-250000000}"
RUN_TIMEOUT_SECONDS="${VGG_CLOSED_SHELL_TIMEOUT_SECONDS:-12000}"

IMG_IDX="0"
IMAGE=""
SHAPE="16x16"
FLOW="os"
LANES="${VGG_CLOSED_LANES:-4}"
CLK_DIV="${VGG_CLOSED_CLK_DIV:-0}"
PPB_DEPTH="${VGG_CLOSED_PPB_DEPTH:-8192}"

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
        --lanes)
            LANES="${2:?missing lanes}"
            shift 2
            ;;
        --clk-div)
            CLK_DIV="${2:?missing clk-div}"
            shift 2
            ;;
        --ppb-depth)
            PPB_DEPTH="${2:?missing ppb-depth}"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [img_idx] [--image <file>] [--shape 4x4|8x8|16x16|8x32] [--flow os|ws] [--lanes 1|2|4] [--clk-div 0|1|2|3] [--ppb-depth <words>]"
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

case "$LANES" in
    1|2|4) ;;
    *)
        echo "Invalid lanes: $LANES (expected 1, 2, or 4)" >&2
        exit 2
        ;;
esac

case "$PPB_DEPTH" in
    ''|*[!0-9]*)
        echo "Invalid ppb-depth: $PPB_DEPTH (expected positive integer)" >&2
        exit 2
        ;;
    0)
        echo "Invalid ppb-depth: 0 (expected positive integer)" >&2
        exit 2
        ;;
esac

GEN_ARGS=()
GEN_ARGS+=(--timeout-cycles "$TIMEOUT_CYCLES")
GEN_ARGS+=(--shape "$SHAPE")
GEN_ARGS+=(--flow "$FLOW")
GEN_ARGS+=(--lanes "$LANES")
GEN_ARGS+=(--clk-div "$CLK_DIV")
GEN_ARGS+=(--ppb-depth "$PPB_DEPTH")
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
timeout "$RUN_TIMEOUT_SECONDS" stdbuf -oL -eL "$OUT_DIR/obj_dir/Vtb_soc_vgg_closed_loop" 2>&1 | tee "$LOG_FILE"
RUN_RC=${PIPESTATUS[0]}
set -e

if [[ "$RUN_RC" -eq 124 ]]; then
    echo "[SHELL_TIMEOUT] ${RUN_TIMEOUT_SECONDS} seconds" | tee -a "$LOG_FILE"
fi

python3 -B "$ROOT/tools/report_perf_summary.py" "$LOG_FILE" >> "$LOG_FILE"

grep -E '\[PASS\]|\[FAIL\]|\[TIMEOUT\]|\[SHELL_TIMEOUT\]|\[PERF\]|\[PERF_SUMMARY\]|^\||Cycles' "$LOG_FILE" || true

if grep -qE '\[FAIL\]|\[TIMEOUT\]|\[SHELL_TIMEOUT\]' "$LOG_FILE"; then
    RUN_RC=1
elif ! grep -q '\[PASS\]' "$LOG_FILE"; then
    RUN_RC=1
fi

PRED=$(grep -oP 'Predicted class: \K\d+' "$LOG_FILE" || true)
if [[ -n "$PRED" ]]; then
    echo "  Predicted: ${CLASSES[$PRED]} (class $PRED)"
fi

exit "$RUN_RC"
