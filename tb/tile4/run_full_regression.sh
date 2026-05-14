#!/bin/bash
# ============================================================================
# NPU Full Regression — All 4 shapes, multiple K values, multi-tile
# Run from: ~/726_NPU/tb/tile4
# ============================================================================
set -e
cd "$(dirname "$0")"

RTL="../../rtl/pe/fp16_mul.v \
     ../../rtl/pe/fp16_add.v \
     ../../rtl/pe/fp32_add.v \
     ../../rtl/pe/pe_top.v \
     ../../rtl/common/fifo.v \
     ../../rtl/common/axi_monitor.v \
     ../../rtl/common/op_counter.v \
     ../../rtl/buf/pingpong_buf.v \
     ../../rtl/buf/psum_out_buf.v \
     ../../rtl/array/reconfig_pe_array.v \
     ../../rtl/power/npu_power.v \
     ../../rtl/ctrl/npu_ctrl.v \
     ../../rtl/axi/npu_axi_lite.v \
     ../../rtl/axi/npu_dma.v \
     ../../rtl/top/npu_top.v"

WFLAGS="-Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-PINMISSING -Wno-INITIALDLY"

run_test() {
    local name=$1
    local expect=$2
    echo "===================================================================="
    echo "  $name  (expect $expect PASS)"
    echo "===================================================================="
    rm -rf obj_dir
    verilator --binary +incdir+${name} --top-module tb_npu_tile_gemm_v2 \
      --timing $WFLAGS $RTL ../tb_npu_tile_gemm_v2.v 2>&1 | tail -1
    ./obj_dir/Vtb_npu_tile_gemm_v2 | grep -E "PASS|FAIL" | head -3
    echo ""
}

echo "==================== NPU FULL REGRESSION (Verilator) ===================="
echo ""

# --- Shape baselines (K=4) ---
run_test test_4x4    "16"
run_test test_8x8    "64"
run_test test_16x16_K4 "256"
run_test test_8x32    "256"

# --- 16x16 K ≠ 4 (SIMD pad) ---
run_test test_16x16_K2  "256"
run_test test_16x16_K3  "256"
run_test test_16x16_K5  "256"
run_test test_16x16_K7  "256"

# --- 16x16 K-split ---
run_test test_16x16_K20 "256"
run_test test_16x16_K21 "256"
run_test test_16x16_K32 "256"
run_test test_16x16_K40 "256"

# --- 16x16 multi-tile ---
run_test test_16x16_M24 "384"
run_test test_16x16_N20 "320"

echo "==================== DONE ===================="
