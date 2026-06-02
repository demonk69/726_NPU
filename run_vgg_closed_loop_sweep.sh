#!/usr/bin/env bash
# Sweep runtime closed-loop VGG across tile shapes and OS/WS dataflows.
#
# Usage:
#   ./run_vgg_closed_loop_sweep.sh
#   ./run_vgg_closed_loop_sweep.sh --image pic/test_cifar10_5.jpeg
#   ./run_vgg_closed_loop_sweep.sh --img-idx 7
#   ./run_vgg_closed_loop_sweep.sh --shapes 4x4,8x8 --flows os,ws

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

SHAPES=(4x4 8x8 16x16 8x32)
FLOWS=(os ws)
IMAGE=""
IMG_IDX="0"
OUT_DIR=""
TIMEOUT_CYCLES=""
STOP_ON_FAIL=0

usage() {
    cat <<'EOF'
Usage: ./run_vgg_closed_loop_sweep.sh [options]

Options:
  --image <file>          Run the sweep on an arbitrary image file.
  --img-idx <idx>         Run the sweep on a CIFAR-10 index when --image is not used. Default: 0.
  --shapes <csv>          Tile shapes to run. Default: 4x4,8x8,16x16,8x32.
  --flows <csv>           Dataflows to run. Default: os,ws.
  --timeout-cycles <n>    Override VGG_CLOSED_TIMEOUT_CYCLES for each case.
  --out-dir <dir>         Result directory. Default: sim/vgg_closed_loop_sweep_<timestamp>.
  --stop-on-fail          Stop after the first non-PASS case.
  --help, -h              Show this help.

Notes:
  This script calls run_vgg_closed_loop.sh serially. That script rebuilds
  sim/vgg_closed_loop for each case, so do not run this sweep concurrently
  with another closed-loop run in the same repo.
EOF
}

split_csv() {
    local raw="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<< "$raw"
}

validate_shapes() {
    local shape
    for shape in "${SHAPES[@]}"; do
        case "$shape" in
            4x4|8x8|16x16|8x32) ;;
            *) echo "Invalid shape: $shape" >&2; exit 2 ;;
        esac
    done
}

validate_flows() {
    local flow
    for flow in "${FLOWS[@]}"; do
        case "$flow" in
            os|ws) ;;
            *) echo "Invalid flow: $flow" >&2; exit 2 ;;
        esac
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            IMAGE="${2:?missing image path}"
            shift 2
            ;;
        --img-idx)
            IMG_IDX="${2:?missing image index}"
            shift 2
            ;;
        --shapes)
            split_csv "${2:?missing shape list}" SHAPES
            shift 2
            ;;
        --flows)
            split_csv "${2:?missing flow list}" FLOWS
            shift 2
            ;;
        --timeout-cycles)
            TIMEOUT_CYCLES="${2:?missing timeout cycles}"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="${2:?missing output directory}"
            shift 2
            ;;
        --stop-on-fail)
            STOP_ON_FAIL=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

validate_shapes
validate_flows

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$ROOT/sim/vgg_closed_loop_sweep_$(date +%Y%m%d_%H%M%S)"
elif [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$ROOT/$OUT_DIR"
fi

mkdir -p "$OUT_DIR/logs"

if [[ -n "$TIMEOUT_CYCLES" ]]; then
    export VGG_CLOSED_TIMEOUT_CYCLES="$TIMEOUT_CYCLES"
fi

SUMMARY_TSV="$OUT_DIR/summary.tsv"
SUMMARY_MD="$OUT_DIR/summary.md"

printf "shape\tflow\tstatus\tpred\texact\tfixed\tcycles\telapsed_s\trc\tlog\n" > "$SUMMARY_TSV"

INPUT_ARGS=()
INPUT_DESC="img_idx=$IMG_IDX"
if [[ -n "$IMAGE" ]]; then
    INPUT_ARGS+=(--image "$IMAGE")
    INPUT_DESC="image=$IMAGE"
else
    INPUT_ARGS+=("$IMG_IDX")
fi

echo "=== Closed-loop sweep ==="
echo "Input:  $INPUT_DESC"
echo "Shapes: ${SHAPES[*]}"
echo "Flows:  ${FLOWS[*]}"
echo "Output: $OUT_DIR"
if [[ -n "${VGG_CLOSED_TIMEOUT_CYCLES:-}" ]]; then
    echo "VGG_CLOSED_TIMEOUT_CYCLES=$VGG_CLOSED_TIMEOUT_CYCLES"
fi

run_case() {
    local shape="$1"
    local flow="$2"
    local case_name="${shape}_${flow}"
    local log_file="$OUT_DIR/logs/${case_name}.log"
    local run_log_copy="$OUT_DIR/logs/${case_name}.run.log"
    local start_s end_s elapsed_s rc status cycles pred exact fixed

    echo
    echo "=== Case: shape=$shape flow=$flow ==="
    start_s=$(date +%s)
    set +e
    "$ROOT/run_vgg_closed_loop.sh" "${INPUT_ARGS[@]}" --shape "$shape" --flow "$flow" 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    end_s=$(date +%s)
    elapsed_s=$((end_s - start_s))

    if [[ -f "$ROOT/sim/vgg_closed_loop/run.log" ]]; then
        cp "$ROOT/sim/vgg_closed_loop/run.log" "$run_log_copy"
    fi

    if grep -q '\[PASS\]' "$log_file"; then
        status="PASS"
    elif grep -q '\[TIMEOUT\]' "$log_file"; then
        status="TIMEOUT"
    elif grep -q '\[FAIL\]' "$log_file"; then
        status="FAIL"
    else
        status="ERROR"
    fi

    cycles=$(grep -oE 'Cycles: [0-9]+' "$log_file" | awk '{print $2}' | tail -n 1 || true)
    pred=$(sed -n 's/.*Predicted class: \([0-9][0-9]*\).*/\1/p' "$log_file" | tail -n 1 || true)
    exact=$(sed -n 's/.*expected exact-python: \([0-9][0-9]*\).*/\1/p' "$log_file" | tail -n 1 || true)
    fixed=$(sed -n 's/.*fixed-runtime: \([0-9][0-9]*\).*/\1/p' "$log_file" | tail -n 1 || true)

    cycles=${cycles:-NA}
    pred=${pred:-NA}
    exact=${exact:-NA}
    fixed=${fixed:-NA}

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$shape" "$flow" "$status" "$pred" "$exact" "$fixed" \
        "$cycles" "$elapsed_s" "$rc" "$log_file" >> "$SUMMARY_TSV"

    echo "[SUMMARY] shape=$shape flow=$flow status=$status pred=$pred exact=$exact fixed=$fixed cycles=$cycles elapsed_s=$elapsed_s rc=$rc"

    if [[ "$status" != "PASS" && "$STOP_ON_FAIL" -eq 1 ]]; then
        return 1
    fi
    return 0
}

overall_rc=0
for shape in "${SHAPES[@]}"; do
    for flow in "${FLOWS[@]}"; do
        if ! run_case "$shape" "$flow"; then
            overall_rc=1
            break 2
        fi
    done
done

{
    echo "# Closed-loop sweep summary"
    echo
    echo "Input: $INPUT_DESC"
    echo
    echo "| shape | flow | status | pred | exact | fixed | cycles | elapsed_s | rc | log |"
    echo "|---|---|---|---:|---:|---:|---:|---:|---:|---|"
    while IFS=$'\t' read -r shape flow status pred exact fixed cycles elapsed_s rc log; do
        [[ "$shape" == "shape" ]] && continue
        printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
            "$shape" "$flow" "$status" "$pred" "$exact" "$fixed" \
            "$cycles" "$elapsed_s" "$rc" "$(basename "$log")"
    done < "$SUMMARY_TSV"
} > "$SUMMARY_MD"

echo
echo "=== Sweep summary ==="
printf "%-6s %-4s %-8s %-5s %-5s %-5s %-12s %-9s %-3s %s\n" \
    "shape" "flow" "status" "pred" "exact" "fixed" "cycles" "elapsed" "rc" "log"
while IFS=$'\t' read -r shape flow status pred exact fixed cycles elapsed_s rc log; do
    [[ "$shape" == "shape" ]] && continue
    printf "%-6s %-4s %-8s %-5s %-5s %-5s %-12s %-9s %-3s %s\n" \
        "$shape" "$flow" "$status" "$pred" "$exact" "$fixed" \
        "$cycles" "$elapsed_s" "$rc" "$(basename "$log")"
done < "$SUMMARY_TSV"

echo
echo "Saved:"
echo "  $SUMMARY_TSV"
echo "  $SUMMARY_MD"
echo "  $OUT_DIR/logs/"

if awk -F '\t' 'NR > 1 && $3 != "PASS" { found = 1 } END { exit found ? 0 : 1 }' "$SUMMARY_TSV"; then
    overall_rc=1
fi

exit "$overall_rc"
