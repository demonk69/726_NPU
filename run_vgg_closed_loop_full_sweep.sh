#!/usr/bin/env bash
# Sweep runtime closed-loop VGG across all PE tile shapes, OS/WS dataflows,
# and INT8 SIMD lane counts. Stops at the first failing case by default.
#
# Usage:
#   ./run_vgg_closed_loop_full_sweep.sh
#   ./run_vgg_closed_loop_full_sweep.sh --img-idx 7
#   ./run_vgg_closed_loop_full_sweep.sh --image pic/test_cifar10_5.jpeg
#   ./run_vgg_closed_loop_full_sweep.sh --lanes 2,4 --shapes 8x8,16x16

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

SHAPES=(4x4 8x8 16x16 8x32)
FLOWS=(os ws)
LANES_LIST=(1 2 4)
IMAGE=""
IMG_IDX="0"
OUT_DIR=""
TIMEOUT_CYCLES="1000000000"
SHELL_TIMEOUT_SECONDS="21600"
STOP_ON_FAIL=1

usage() {
    cat <<'EOF'
Usage: ./run_vgg_closed_loop_full_sweep.sh [options]

Options:
  --image <file>             Run on an arbitrary image file.
  --img-idx <idx>            Run on a CIFAR-10 index when --image is not used. Default: 0.
  --shapes <csv>             PE tile shapes. Default: 4x4,8x8,16x16,8x32.
  --flows <csv>              Dataflows. Default: os,ws.
  --lanes <csv>              INT8 SIMD lanes. Default: 1,2,4.
  --timeout-cycles <n>       Per-case RTL timeout cycles. Default: 1000000000.
  --shell-timeout-seconds <n> Per-case host timeout seconds. Default: 21600.
  --out-dir <dir>            Result directory. Default: sim/vgg_closed_loop_full_sweep_<timestamp>.
  --continue-on-fail         Do not stop at the first failing case.
  --help, -h                 Show this help.

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

validate_lanes() {
    local lanes
    for lanes in "${LANES_LIST[@]}"; do
        case "$lanes" in
            1|2|4) ;;
            *) echo "Invalid lanes: $lanes" >&2; exit 2 ;;
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
        --lanes)
            split_csv "${2:?missing lanes list}" LANES_LIST
            shift 2
            ;;
        --timeout-cycles)
            TIMEOUT_CYCLES="${2:?missing timeout cycles}"
            shift 2
            ;;
        --shell-timeout-seconds)
            SHELL_TIMEOUT_SECONDS="${2:?missing shell timeout seconds}"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="${2:?missing output directory}"
            shift 2
            ;;
        --continue-on-fail)
            STOP_ON_FAIL=0
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
validate_lanes

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$ROOT/sim/vgg_closed_loop_full_sweep_$(date +%Y%m%d_%H%M%S)"
elif [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$ROOT/$OUT_DIR"
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/meta"

SUMMARY_TSV="$OUT_DIR/summary.tsv"
SUMMARY_MD="$OUT_DIR/summary.md"

printf "lanes\tshape\tflow\tstatus\tpred\texact\tfixed\tcycles\telapsed_s\trc\tlog\n" > "$SUMMARY_TSV"

INPUT_ARGS=()
INPUT_DESC="img_idx=$IMG_IDX"
if [[ -n "$IMAGE" ]]; then
    INPUT_ARGS+=(--image "$IMAGE")
    INPUT_DESC="image=$IMAGE"
else
    INPUT_ARGS+=("$IMG_IDX")
fi

write_summary_md() {
    {
        echo "# Closed-loop Full Sweep Summary"
        echo
        echo "Input: $INPUT_DESC"
        echo
        echo "Timeout cycles: $TIMEOUT_CYCLES"
        echo
        echo "Shell timeout seconds: $SHELL_TIMEOUT_SECONDS"
        echo
        echo "| lanes | shape | flow | status | pred | exact | fixed | cycles | elapsed_s | rc | log |"
        echo "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|"
        while IFS=$'\t' read -r lanes shape flow status pred exact fixed cycles elapsed_s rc log; do
            [[ "$lanes" == "lanes" ]] && continue
            printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
                "$lanes" "$shape" "$flow" "$status" "$pred" "$exact" "$fixed" \
                "$cycles" "$elapsed_s" "$rc" "$(basename "$log")"
        done < "$SUMMARY_TSV"
    } > "$SUMMARY_MD"
}

print_summary() {
    echo
    echo "=== Sweep summary table ==="
    echo "| lanes | shape | flow | status | pred | exact | fixed | cycles | elapsed_s | rc | log |"
    echo "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|"
    while IFS=$'\t' read -r lanes shape flow status pred exact fixed cycles elapsed_s rc log; do
        [[ "$lanes" == "lanes" ]] && continue
        printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
            "$lanes" "$shape" "$flow" "$status" "$pred" "$exact" "$fixed" \
            "$cycles" "$elapsed_s" "$rc" "$(basename "$log")"
    done < "$SUMMARY_TSV"
    echo
    echo "Saved:"
    echo "  $SUMMARY_TSV"
    echo "  $SUMMARY_MD"
    echo "  $OUT_DIR/logs/"
    echo "  $OUT_DIR/meta/"
}

run_case() {
    local lanes="$1"
    local shape="$2"
    local flow="$3"
    local case_name="L${lanes}_${shape}_${flow}"
    local log_file="$OUT_DIR/logs/${case_name}.log"
    local run_log_copy="$OUT_DIR/logs/${case_name}.run.log"
    local meta_copy="$OUT_DIR/meta/${case_name}.metadata.json"
    local params_copy="$OUT_DIR/meta/${case_name}.params.vh"
    local start_s end_s elapsed_s rc status cycles pred exact fixed

    echo
    echo "=== Case: lanes=$lanes shape=$shape flow=$flow ==="
    start_s=$(date +%s)
    set +e
    VGG_CLOSED_TIMEOUT_CYCLES="$TIMEOUT_CYCLES" \
    VGG_CLOSED_SHELL_TIMEOUT_SECONDS="$SHELL_TIMEOUT_SECONDS" \
        "$ROOT/run_vgg_closed_loop.sh" "${INPUT_ARGS[@]}" \
            --shape "$shape" --flow "$flow" --lanes "$lanes" 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    end_s=$(date +%s)
    elapsed_s=$((end_s - start_s))

    if [[ -f "$ROOT/sim/vgg_closed_loop/run.log" ]]; then
        cp "$ROOT/sim/vgg_closed_loop/run.log" "$run_log_copy"
    fi
    if [[ -f "$ROOT/sim/vgg_closed_loop/metadata.json" ]]; then
        cp "$ROOT/sim/vgg_closed_loop/metadata.json" "$meta_copy"
    fi
    if [[ -f "$ROOT/sim/vgg_closed_loop/soc_vgg_closed_loop_params.vh" ]]; then
        cp "$ROOT/sim/vgg_closed_loop/soc_vgg_closed_loop_params.vh" "$params_copy"
    fi

    if grep -q '\[PASS\]' "$log_file"; then
        status="PASS"
    elif grep -q '\[TIMEOUT\]' "$log_file"; then
        status="TIMEOUT"
    elif grep -q '\[SHELL_TIMEOUT\]' "$log_file"; then
        status="SHELL_TIMEOUT"
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

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$lanes" "$shape" "$flow" "$status" "$pred" "$exact" "$fixed" \
        "$cycles" "$elapsed_s" "$rc" "$log_file" >> "$SUMMARY_TSV"

    echo "[SUMMARY] lanes=$lanes shape=$shape flow=$flow status=$status pred=$pred exact=$exact fixed=$fixed cycles=$cycles elapsed_s=$elapsed_s rc=$rc"

    [[ "$status" == "PASS" && "$rc" -eq 0 ]]
}

echo "=== Closed-loop full sweep ==="
echo "Input:  $INPUT_DESC"
echo "Lanes:  ${LANES_LIST[*]}"
echo "Shapes: ${SHAPES[*]}"
echo "Flows:  ${FLOWS[*]}"
echo "Output: $OUT_DIR"
echo "VGG_CLOSED_TIMEOUT_CYCLES=$TIMEOUT_CYCLES"
echo "VGG_CLOSED_SHELL_TIMEOUT_SECONDS=$SHELL_TIMEOUT_SECONDS"
echo "Stop on fail: $STOP_ON_FAIL"

overall_rc=0
for lanes in "${LANES_LIST[@]}"; do
    for shape in "${SHAPES[@]}"; do
        for flow in "${FLOWS[@]}"; do
            if ! run_case "$lanes" "$shape" "$flow"; then
                overall_rc=1
                write_summary_md
                print_summary
                if [[ "$STOP_ON_FAIL" -eq 1 ]]; then
                    echo "[STOP] First failing case: lanes=$lanes shape=$shape flow=$flow" >&2
                    exit 1
                fi
            fi
        done
    done
done

write_summary_md
print_summary

exit "$overall_rc"
