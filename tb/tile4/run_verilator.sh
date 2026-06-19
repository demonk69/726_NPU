#!/bin/bash
# =============================================================================
# run_verilator.sh — Verilator verification for NPU tile-mode GEMM
#
# Usage:
#   ./run_verilator.sh --shape 16x16 --M 16 --K 4 --N 16
#   ./run_verilator.sh --shape 8x32  --M 8  --K 2 --N 32 --bias
#   ./run_verilator.sh --shape 16x16 --M 16 --K 5 --N 16 --lanes 2 --flow ws
#   ./run_verilator.sh --all              # full regression
#   ./run_verilator.sh --all --verilator  # Verilator full regression
#   ./run_verilator.sh --all --icarus     # Icarus full regression
#
# Prerequisites: verilator 5.030+, iverilog 11.0+, python3
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPU_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RTL_DIR="$NPU_ROOT/rtl"
TB_DIR="$NPU_ROOT/tb"
GEN_PY="$SCRIPT_DIR/gen_multi_shape_data.py"
WORK_DIR="${WORK_DIR:-/tmp/opencode/verilator_regress}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
CLEANUP_WORK="${CLEANUP_WORK:-0}"

# ── default test matrix ──
DEFAULT_SHAPES=(
    "4x4:4:4:4"
    "8x8:8:8:8"
    "16x16:16:4:16"
    "8x32:8:2:32"
)

# ── helper ──
log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

# ── RTL file list (excludes soc_top which needs picorv32) ──
RTL_FILES=(
    "$RTL_DIR/pe/fp16_add.v" "$RTL_DIR/pe/fp16_mul.v" "$RTL_DIR/pe/fp32_add.v"
    "$RTL_DIR/pe/pe_top.v"
    "$RTL_DIR/common/fifo.v" "$RTL_DIR/common/op_counter.v" "$RTL_DIR/common/axi_monitor.v"
    "$RTL_DIR/buf/pingpong_buf.v" "$RTL_DIR/buf/psum_out_buf.v"
    "$RTL_DIR/array/reconfig_pe_array.v"
    "$RTL_DIR/axi/npu_axi_lite.v" "$RTL_DIR/axi/npu_dma.v"
    "$RTL_DIR/ctrl/npu_ctrl.v"
    "$RTL_DIR/power/npu_power.v"
    "$RTL_DIR/top/npu_top.v"
)
TB_FILE="$TB_DIR/tb_npu_tile_gemm_v2.v"

# ── verilator warnings to suppress ──
VL_WARN="-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSED -Wno-UNDRIVEN \
         -Wno-UNOPTFLAT -Wno-PINMISSING -Wno-CASEINCOMPLETE \
         -Wno-COMBDLY -Wno-INITIALDLY -Wno-LITENDIAN"

run_icarus() {
    local test_dir="$1" test_name="$2"
    local timeout_sec="${3:-$TIMEOUT_SEC}"
    log "ICARUS  $test_name"
    local vvp="$WORK_DIR/${test_name}/sim_ivl.vvp"
    mkdir -p "$(dirname "$vvp")"
    iverilog -g2012 -I "$test_dir" -o "$vvp" \
        "${RTL_FILES[@]}" "$TB_FILE" 2>&1 || die "icarus compile fail: $test_name"
    (cd "$test_dir" && timeout "$timeout_sec" vvp "$vvp" 2>&1 | grep -E 'PASS|FAIL|WARN|^RESULT')
}

run_verilator() {
    local test_dir="$1" test_name="$2"
    local timeout_sec="${3:-$TIMEOUT_SEC}"
    log "VERILATOR $test_name"
    local vdir="$WORK_DIR/${test_name}/obj_dir"
    rm -rf "$vdir"
    mkdir -p "$vdir"
    verilator --binary --timing \
        -I"$test_dir" \
        -Mdir "$vdir" \
        --top-module tb_npu_tile_gemm_v2 \
        $VL_WARN \
        +define+VERILATOR_TRACE \
        "${RTL_FILES[@]}" "$TB_FILE" 2>&1 || die "verilator compile fail: $test_name"
    (cd "$test_dir" && timeout "$timeout_sec" "$vdir/Vtb_npu_tile_gemm_v2" 2>&1 | grep -E 'PASS|FAIL|WARN|^RESULT')
}

run_test() {
    local shape="$1" M="$2" K="$3" N="$4"
    local extra_flags="${5:-}"
    local sim_mode="${6:-icarus}"
    local lanes="${7:-4}"
    local perf_only="${8:-0}"
    local timeout_sec="${9:-$TIMEOUT_SEC}"
    local strict_aw="${10:-0}"
    local flow="${11:-os}"
    local bias_flag=""
    local perf_flag=""
    local strict_aw_flag=""
    local tag="${shape}_${flow}_L${lanes}_M${M}_K${K}_N${N}"
    [[ "$extra_flags" == *"--bias"* ]] && bias_flag="--bias" && tag="${tag}_bias"
    [[ "$extra_flags" == *"--activation"* ]] && tag="${tag}_act"
    [[ "$perf_only" == "1" ]] && perf_flag="--perf-only" && tag="${tag}_perf"
    [[ "$strict_aw" == "1" ]] && strict_aw_flag="--strict-aw"

    local out_dir="$WORK_DIR/${tag}"
    log "GEN  $tag"
    python3 "$GEN_PY" --shape "$shape" --M "$M" --K "$K" --N "$N" \
        --lanes "$lanes" --flow "$flow" $bias_flag $perf_flag $strict_aw_flag \
        --out-dir "$out_dir" --name "$tag" 2>&1

    local data_dir="$out_dir/$tag"
    case "$sim_mode" in
        icarus)    run_icarus "$data_dir" "$tag" "$timeout_sec" ;;
        verilator) run_verilator "$data_dir" "$tag" "$timeout_sec" ;;
    esac
    if [[ "$CLEANUP_WORK" == "1" ]]; then
        rm -rf "$out_dir"
    fi
}

run_all() {
    local mode="$1"
    local lanes="${2:-4}"
    local perf_only="${3:-0}"
    local timeout_sec="${4:-$TIMEOUT_SEC}"
    local strict_aw="${5:-0}"
    local flow="${6:-os}"
    local total=0 pass=0

    log "=== FULL REGRESSION ($mode flow=$flow) ==="

    for entry in "${DEFAULT_SHAPES[@]}"; do
        IFS=':' read -r shape M K N <<< "$entry"
        run_test "$shape" "$M" "$K" "$N" "" "$mode" "$lanes" "$perf_only" "$timeout_sec" "$strict_aw" "$flow"
        run_test "$shape" "$M" "$K" "$N" "--bias" "$mode" "$lanes" "$perf_only" "$timeout_sec" "$strict_aw" "$flow"
    done

    # K-split tests (multiples of 4 — known to work)
    for entry in "16x16:16:20:16" "16x16:16:32:16" "8x32:8:5:32"; do
        IFS=':' read -r shape M K N <<< "$entry"
        run_test "$shape" "$M" "$K" "$N" "" "$mode" "$lanes" "$perf_only" "$timeout_sec" "$strict_aw" "$flow"
    done

    log "=== REGRESSION COMPLETE ==="
}

# ── parse args ──
VERILATOR_ONLY=0
ICARUS_ONLY=0
ALL=0
LANES_VAL=4
PERF_ONLY=0
TIMEOUT_ARG="$TIMEOUT_SEC"
STRICT_AW=0
FLOW_VAL=os

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)       ALL=1 ;;
        --verilator) VERILATOR_ONLY=1 ;;
        --icarus)    ICARUS_ONLY=1 ;;
        --shape)     SHAPE="$2"; shift ;;
        --M)         M_VAL="$2"; shift ;;
        --K)         K_VAL="$2"; shift ;;
        --N)         N_VAL="$2"; shift ;;
        --lanes)     LANES_VAL="$2"; shift ;;
        --flow)      FLOW_VAL="$2"; shift ;;
        --perf-only) PERF_ONLY=1 ;;
        --timeout-sec) TIMEOUT_ARG="$2"; shift ;;
        --strict-aw) STRICT_AW=1 ;;
        --bias)      BIAS_FLAG="--bias" ;;
        --help|-h)
            echo "Usage: $0 [--all] [--verilator|--icarus] [--shape S --M M --K K --N N] [--flow os|ws] [--lanes 1|2|4] [--bias] [--perf-only] [--timeout-sec SEC] [--strict-aw]"
            exit 0 ;;
        *) die "Unknown arg: $1" ;;
    esac
    shift
done

if [[ "$ALL" -eq 1 ]]; then
    [[ "$LANES_VAL" =~ ^(1|2|4)$ ]] || die "--lanes must be 1, 2, or 4"
    [[ "$FLOW_VAL" =~ ^(os|ws)$ ]] || die "--flow must be os or ws"
    if [[ "$VERILATOR_ONLY" -eq 1 ]]; then
        run_all "verilator" "$LANES_VAL" "$PERF_ONLY" "$TIMEOUT_ARG" "$STRICT_AW" "$FLOW_VAL"
    elif [[ "$ICARUS_ONLY" -eq 1 ]]; then
        run_all "icarus" "$LANES_VAL" "$PERF_ONLY" "$TIMEOUT_ARG" "$STRICT_AW" "$FLOW_VAL"
    else
        run_all "icarus" "$LANES_VAL" "$PERF_ONLY" "$TIMEOUT_ARG" "$STRICT_AW" "$FLOW_VAL"
        echo ""
        run_all "verilator" "$LANES_VAL" "$PERF_ONLY" "$TIMEOUT_ARG" "$STRICT_AW" "$FLOW_VAL"
    fi
elif [[ -n "$SHAPE" && -n "$M_VAL" && -n "$K_VAL" && -n "$N_VAL" ]]; then
    [[ "$LANES_VAL" =~ ^(1|2|4)$ ]] || die "--lanes must be 1, 2, or 4"
    [[ "$FLOW_VAL" =~ ^(os|ws)$ ]] || die "--flow must be os or ws"
    if [[ "$VERILATOR_ONLY" -eq 1 ]]; then
        run_test "$SHAPE" "$M_VAL" "$K_VAL" "$N_VAL" "$BIAS_FLAG" "verilator" "$LANES_VAL" "$PERF_ONLY" "$TIMEOUT_ARG" "$STRICT_AW" "$FLOW_VAL"
    elif [[ "$ICARUS_ONLY" -eq 1 ]]; then
        run_test "$SHAPE" "$M_VAL" "$K_VAL" "$N_VAL" "$BIAS_FLAG" "icarus" "$LANES_VAL" "$PERF_ONLY" "$TIMEOUT_ARG" "$STRICT_AW" "$FLOW_VAL"
    else
        run_test "$SHAPE" "$M_VAL" "$K_VAL" "$N_VAL" "$BIAS_FLAG" "icarus" "$LANES_VAL" "$PERF_ONLY" "$TIMEOUT_ARG" "$STRICT_AW" "$FLOW_VAL"
        echo ""
        run_test "$SHAPE" "$M_VAL" "$K_VAL" "$N_VAL" "$BIAS_FLAG" "verilator" "$LANES_VAL" "$PERF_ONLY" "$TIMEOUT_ARG" "$STRICT_AW" "$FLOW_VAL"
    fi
else
    die "Specify --all or --shape S --M M --K K --N N"
fi
