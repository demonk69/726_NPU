#!/bin/bash
# ============================================================================
# NPU Regression: All 4 shapes (4x4, 8x8, 16x16, 8x32) on Verilator
# Run from: /home/lab_726/726_NPU/tb/tile4
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
    echo ""
    echo "===================================================================="
    echo "  $name"
    echo "===================================================================="
    verilator --binary +incdir+${name} --top-module tb_npu_tile_gemm_v2 \
      --timing $WFLAGS $RTL ../tb_npu_tile_gemm_v2.v
    ./obj_dir/Vtb_npu_tile_gemm_v2 | grep -E "PASS|FAIL|CHECKS"
}

run_test test_4x4
run_test test_8x8
run_test test_16x16
run_test test_8x32
