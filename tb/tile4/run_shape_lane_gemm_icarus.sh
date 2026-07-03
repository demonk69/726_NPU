#!/usr/bin/env bash
# =============================================================================
# run_shape_lane_gemm_icarus.sh — Icarus GEMM regression for every tile shape
# and every supported INT8 SIMD lane count.
#
# Default matrix:
#   shapes: 4x4, 8x8, 16x16, 8x32
#   lanes:  1, 2, 4, 8
#
# Generated data and compiled vvp files are created under WORK_DIR and removed
# automatically unless KEEP_WORK=1 is set.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPU_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RTL_DIR="$NPU_ROOT/rtl"
TB_DIR="$NPU_ROOT/tb"
GEN_PY="$SCRIPT_DIR/gen_multi_shape_data.py"
WORK_DIR="${WORK_DIR:-/tmp/opencode/shape_lane_gemm_icarus_$$}"
KEEP_WORK="${KEEP_WORK:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"

RTL_FILES=(
    "$RTL_DIR/pe/fp16_add.v"
    "$RTL_DIR/pe/fp16_mul.v"
    "$RTL_DIR/pe/fp32_add.v"
    "$RTL_DIR/pe/pe_top.v"
    "$RTL_DIR/array/reconfig_pe_array.v"
    "$RTL_DIR/buf/pingpong_buf.v"
    "$RTL_DIR/buf/psum_out_buf.v"
    "$RTL_DIR/common/fifo.v"
    "$RTL_DIR/common/axi_monitor.v"
    "$RTL_DIR/common/op_counter.v"
    "$RTL_DIR/power/npu_power.v"
    "$RTL_DIR/axi/npu_dma.v"
    "$RTL_DIR/axi/npu_axi_lite.v"
    "$RTL_DIR/ctrl/npu_ctrl.v"
    "$RTL_DIR/top/npu_top.v"
)
TB_FILE="$TB_DIR/tb_npu_tile_gemm_v2.v"

SHAPE_CASES=(
    "4x4:4:4:4"
    "8x8:8:4:8"
    "16x16:16:4:16"
    "8x32:8:4:32"
)
LANES_LIST=(1 2 4 8)

cleanup() {
    if [[ "$KEEP_WORK" != "1" ]]; then
        rm -rf "$WORK_DIR"
    else
        echo "[INFO] Keeping WORK_DIR=$WORK_DIR"
    fi
}
trap cleanup EXIT

log() {
    printf '[%(%H:%M:%S)T] %s\n' -1 "$*"
}

run_case() {
    local shape="$1"
    local m_dim="$2"
    local k_dim="$3"
    local n_dim="$4"
    local lanes="$5"
    local name="shape_${shape}_L${lanes}_M${m_dim}_K${k_dim}_N${n_dim}"
    local out_dir="$WORK_DIR/$name"
    local case_dir="$out_dir/$name"
    local vvp="$out_dir/sim.vvp"
    local log_file="$out_dir/run.log"

    log "GEN     $name"
    python3 "$GEN_PY" \
        --shape "$shape" \
        --M "$m_dim" \
        --K "$k_dim" \
        --N "$n_dim" \
        --lanes "$lanes" \
        --out-dir "$out_dir" \
        --name "$name" >/dev/null

    log "COMPILE $name"
    iverilog -g2012 -I "$case_dir" -o "$vvp" \
        "${RTL_FILES[@]}" "$TB_FILE"

    log "RUN     $name"
    if ! timeout "$TIMEOUT_SEC" vvp "$vvp" >"$log_file" 2>&1; then
        echo "[FAIL] $name"
        sed -n '/FAIL\|WARN\|RESULT\|PASS/p' "$log_file"
        return 1
    fi

    if ! grep -q "\[PASS\]" "$log_file"; then
        echo "[FAIL] $name did not report PASS"
        sed -n '/FAIL\|WARN\|RESULT\|PASS/p' "$log_file"
        return 1
    fi

    sed -n '/RESULT\|\[PASS\]\|\[WARN\]/p' "$log_file"
}

main() {
    mkdir -p "$WORK_DIR"
    log "WORK_DIR=$WORK_DIR"

    local total=0
    local passed=0
    local failed=0

    for lanes in "${LANES_LIST[@]}"; do
        for entry in "${SHAPE_CASES[@]}"; do
            IFS=':' read -r shape m_dim k_dim n_dim <<< "$entry"
            total=$((total + 1))
            if run_case "$shape" "$m_dim" "$k_dim" "$n_dim" "$lanes"; then
                passed=$((passed + 1))
            else
                failed=$((failed + 1))
            fi
        done
    done

    echo "SUMMARY total=$total passed=$passed failed=$failed"
    if [[ "$failed" -ne 0 ]]; then
        exit 1
    fi
}

main "$@"
