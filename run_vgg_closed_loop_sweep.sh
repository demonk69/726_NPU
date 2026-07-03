#!/usr/bin/env bash
# Sweep runtime closed-loop VGG across tile shapes, flows, lanes, cores, CLK_DIV,
# and ping-pong buffer depth.
#
# Usage examples:
#   ./run_vgg_closed_loop_sweep.sh --lanes 8 --num-cores 1,2
#       → lanes=8, all shapes×flows×cores(1,2)
#   ./run_vgg_closed_loop_sweep.sh --num-cores 2 --shapes 16x16 --flows os,ws
#       → cores=2, shapes=16x16, both flows, default lanes=8
#   ./run_vgg_closed_loop_sweep.sh --clk-divs 0,1,2 --ppb-depths 1024,4096,8192 --shapes 16x16 --lanes 8
#       → DFS + buffer-depth sweep on 16x16, all flows, cores=1, lanes=8

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

SHAPES=(4x4 8x8 16x16 8x32)
FLOWS=(os ws)
LANES_LIST=(8)
NUM_CORES_LIST=(1)
CLK_DIVS=(0)
PPB_DEPTHS=(8192)
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
  --img-idx <idx>         Run the sweep on a CIFAR-10 index (default: 0).
  --shapes <csv>          Tile shapes.      Default: 4x4,8x8,16x16,8x32
  --flows <csv>           Dataflows.        Default: os,ws
  --lanes <csv>           INT8 SIMD lanes.  Default: 8
  --num-cores <csv>       Number of cores (1 uses single-core runner). Default: 1
  --clk-divs <csv>        CLK_DIV divisors. Default: 0
  --ppb-depths <csv>      Ping-pong buffer depth in 32-bit words. Default: 8192
  --timeout-cycles <n>    Override VGG_CLOSED_TIMEOUT_CYCLES.
  --out-dir <dir>         Result directory. Default: sim/vgg_closed_loop_sweep_<timestamp>
  --stop-on-fail          Stop after the first non-PASS case.
  --help, -h              Show this help.

Notes:
  This script calls run_vgg_closed_loop.sh serially.
  Each case rebuilds sim/vgg_closed_loop (or sim/vgg_mc_closed_loop), so do not
  run concurrently with another closed-loop run in the same repo.
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
            1|2|4|8) ;;
            *) echo "Invalid lanes: $lanes" >&2; exit 2 ;;
        esac
    done
}

validate_num_cores() {
    local nc
    for nc in "${NUM_CORES_LIST[@]}"; do
        case "$nc" in
            1|2|4) ;;
            *) echo "Invalid num-cores: $nc (1, 2, or 4)" >&2; exit 2 ;;
        esac
    done
}

validate_clk_divs() {
    local div
    for div in "${CLK_DIVS[@]}"; do
        case "$div" in
            0|1|2|3) ;;
            *) echo "Invalid clk-div: $div" >&2; exit 2 ;;
        esac
    done
}

validate_ppb_depths() {
    local depth
    for depth in "${PPB_DEPTHS[@]}"; do
        case "$depth" in
            ''|*[!0-9]*) echo "Invalid ppb-depth: $depth (positive integer required)" >&2; exit 2 ;;
            0) echo "Invalid ppb-depth: 0 (positive integer required)" >&2; exit 2 ;;
        esac
    done
}

summary_value() {
    local log_file="$1"
    local key="$2"
    local value
    value=$(awk -F'|' -v key="$key" '
        NF >= 3 {
            k = $2; v = $3;
            gsub(/^[ \t]+|[ \t]+$/, "", k);
            gsub(/^[ \t]+|[ \t]+$/, "", v);
            if (k == key || k == key "_sum") last = v;
        }
        END { if (last != "") print last; }
    ' "$log_file")
    if [[ -n "$value" ]]; then
        printf "%s" "$value"
    else
        printf "NA"
    fi
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
        --num-cores)
            split_csv "${2:?missing num-cores list}" NUM_CORES_LIST
            shift 2
            ;;
        --clk-divs)
            split_csv "${2:?missing clk-div list}" CLK_DIVS
            shift 2
            ;;
        --ppb-depths)
            split_csv "${2:?missing ppb-depth list}" PPB_DEPTHS
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
validate_lanes
validate_num_cores
validate_clk_divs
validate_ppb_depths

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

printf "shape\tflow\tlanes\tcores\tclk_div\tppb_depth\tstatus\tpred\texact\tfixed\tcycles\tbusy_cycles\tcompute_cycles\tops_per_cycle\tpeak_tops\trd_burst_util\twr_burst_util\telapsed_s\trc\tlog\n" > "$SUMMARY_TSV"

INPUT_ARGS=()
INPUT_DESC="img_idx=$IMG_IDX"
if [[ -n "$IMAGE" ]]; then
    INPUT_ARGS+=(--image "$IMAGE")
    INPUT_DESC="image=$IMAGE"
else
    INPUT_ARGS+=("$IMG_IDX")
fi

echo "=== Closed-loop sweep ==="
echo "Input:    $INPUT_DESC"
echo "Shapes:   ${SHAPES[*]}"
echo "Flows:    ${FLOWS[*]}"
echo "Lanes:    ${LANES_LIST[*]}"
echo "Cores:    ${NUM_CORES_LIST[*]}"
echo "CLK_DIVs: ${CLK_DIVS[*]}"
echo "PPB:      ${PPB_DEPTHS[*]}"
echo "Output:   $OUT_DIR"
if [[ -n "${VGG_CLOSED_TIMEOUT_CYCLES:-}" ]]; then
    echo "VGG_CLOSED_TIMEOUT_CYCLES=$VGG_CLOSED_TIMEOUT_CYCLES"
fi

TOTAL_CASES=$(( ${#SHAPES[@]} * ${#FLOWS[@]} * ${#LANES_LIST[@]} * ${#NUM_CORES_LIST[@]} * ${#CLK_DIVS[@]} * ${#PPB_DEPTHS[@]} ))
echo "Cases:    $TOTAL_CASES"

run_case() {
    local case_idx="$1"
    local total_cases="$2"
    local shape="$3"
    local flow="$4"
    local lanes="$5"
    local nc="$6"
    local clk_div="$7"
    local ppb_depth="$8"

    local runner="$ROOT/run_vgg_closed_loop.sh"
    local run_dir="$ROOT/sim/vgg_closed_loop"
    [[ "$nc" -ne 1 ]] && run_dir="$ROOT/sim/vgg_mc_closed_loop"
    local runner_args=(
        "${INPUT_ARGS[@]}"
        --shape "$shape" --flow "$flow" --lanes "$lanes"
        --num-cores "$nc" --clk-div "$clk_div"
        --ppb-depth "$ppb_depth"
    )
    local case_name="${shape}_${flow}_L${lanes}_C${nc}_D${clk_div}_P${ppb_depth}"
    local progress="[$case_idx/$total_cases]"
    local log_file="$OUT_DIR/logs/${case_name}.log"
    local run_log_copy="$OUT_DIR/logs/${case_name}.run.log"
    local start_s end_s elapsed_s rc status cycles busy_cycles compute_cycles ops_per_cycle perf_summary peak_tops rd_burst_util wr_burst_util

    echo
    echo "$progress === Case: shape=$shape flow=$flow lanes=$lanes cores=$nc clk_div=$clk_div ppb_depth=$ppb_depth runner=$(basename "$runner") ==="
    start_s=$(date +%s)
    set +e
    "$runner" "${runner_args[@]}" 2>&1 | awk -v p="$progress" '{ print p " " $0; fflush(); }' | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    end_s=$(date +%s)
    elapsed_s=$((end_s - start_s))

    if [[ -f "$run_dir/run.log" ]]; then
        cp "$run_dir/run.log" "$run_log_copy"
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
    busy_cycles=$(summary_value "$log_file" "busy_cycles")
    compute_cycles=$(summary_value "$log_file" "compute_cycles")
    ops_per_cycle=$(summary_value "$log_file" "ops_per_cycle")
    peak_tops=$(summary_value "$log_file" "peak_tops")
    rd_burst_util=$(summary_value "$log_file" "rd_burst_util")
    wr_burst_util=$(summary_value "$log_file" "wr_burst_util")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$shape" "$flow" "$lanes" "$nc" "$clk_div" "$ppb_depth" "$status" "$pred" "$exact" "$fixed" \
        "$cycles" "$busy_cycles" "$compute_cycles" "$ops_per_cycle" \
        "$peak_tops" "$rd_burst_util" "$wr_burst_util" \
        "$elapsed_s" "$rc" "$log_file" >> "$SUMMARY_TSV"

    echo "$progress [SUMMARY] shape=$shape flow=$flow lanes=$lanes cores=$nc clk_div=$clk_div ppb_depth=$ppb_depth status=$status pred=$pred exact=$exact fixed=$fixed cycles=$cycles busy=$busy_cycles compute=$compute_cycles ops_per_cycle=$ops_per_cycle peak_tops=$peak_tops rd_burst_util=$rd_burst_util wr_burst_util=$wr_burst_util elapsed_s=$elapsed_s rc=$rc"

    rm -rf "$run_dir"

    if [[ "$status" != "PASS" && "$STOP_ON_FAIL" -eq 1 ]]; then
        return 1
    fi
    return 0
}

overall_rc=0
case_idx=0
for shape in "${SHAPES[@]}"; do
    for flow in "${FLOWS[@]}"; do
        for lanes in "${LANES_LIST[@]}"; do
            for nc in "${NUM_CORES_LIST[@]}"; do
                for clk_div in "${CLK_DIVS[@]}"; do
                    for ppb_depth in "${PPB_DEPTHS[@]}"; do
                        case_idx=$((case_idx + 1))
                        if ! run_case "$case_idx" "$TOTAL_CASES" "$shape" "$flow" "$lanes" "$nc" "$clk_div" "$ppb_depth"; then
                            overall_rc=1
                            break 6
                        fi
                    done
                done
            done
        done
    done
done

{
    echo "# Closed-loop sweep summary"
    echo
    echo "Input: $INPUT_DESC"
    echo
    echo "| shape | flow | lanes | cores | clk_div | ppb_depth | status | pred | exact | fixed | cycles | busy | compute | ops/cycle | TOPS | rd bus util | wr bus util | elapsed_s | rc | log |"
    echo "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    while IFS=$'\t' read -r shape flow lanes nc clk_div ppb_depth status pred exact fixed cycles busy_cycles compute_cycles ops_per_cycle peak_tops rd_burst_util wr_burst_util elapsed_s rc log; do
        [[ "$shape" == "shape" ]] && continue
        printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
            "$shape" "$flow" "$lanes" "$nc" "$clk_div" "$ppb_depth" "$status" "$pred" "$exact" "$fixed" \
            "$cycles" "$busy_cycles" "$compute_cycles" "$ops_per_cycle" \
            "$peak_tops" "$rd_burst_util" "$wr_burst_util" \
            "$elapsed_s" "$rc" "$(basename "$log")"
    done < "$SUMMARY_TSV"
} > "$SUMMARY_MD"

echo
echo "=== Sweep summary ==="
printf "%-6s %-4s %-5s %-5s %-7s %-9s %-8s %-5s %-5s %-5s %-12s %-10s %-10s %-10s %-10s %-12s %-12s %-8s %-3s %s\n" \
    "shape" "flow" "lanes" "cores" "clkdiv" "ppb" "status" "pred" "exact" "fixed" "cycles" "busy" "compute" "ops/cyc" "peak_tops" "rd_bus_util" "wr_bus_util" "elapsed" "rc" "log"

while IFS=$'\t' read -r shape flow lanes nc clk_div ppb_depth status pred exact fixed cycles busy_cycles compute_cycles ops_per_cycle peak_tops rd_burst_util wr_burst_util elapsed_s rc log; do
    [[ "$shape" == "shape" ]] && continue
    printf "%-6s %-4s %-5s %-5s %-7s %-9s %-8s %-5s %-5s %-5s %-12s %-10s %-10s %-10s %-10s %-12s %-12s %-8s %-3s %s\n" \
        "$shape" "$flow" "$lanes" "$nc" "$clk_div" "$ppb_depth" "$status" "$pred" "$exact" "$fixed" \
        "$cycles" "$busy_cycles" "$compute_cycles" "$ops_per_cycle" \
        "$peak_tops" "$rd_burst_util" "$wr_burst_util" \
        "$elapsed_s" "$rc" "$(basename "$log")"
done < "$SUMMARY_TSV"

echo
echo "Saved:"
echo "  $SUMMARY_TSV"
echo "  $SUMMARY_MD"
echo "  $OUT_DIR/logs/"

if awk -F '\t' 'NR > 1 && $7 != "PASS" { found = 1 } END { exit found ? 0 : 1 }' "$SUMMARY_TSV"; then
    overall_rc=1
fi

exit "$overall_rc"
