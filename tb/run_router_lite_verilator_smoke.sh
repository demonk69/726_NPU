#!/usr/bin/env bash
# Verilator smoke for the optional router-lite PE-array path.
#
# Defaults to a true-scale 16x16 npu_top instantiation with USE_ROUTER_MESH=1.
# The testbench is intentionally idle/reset-only at top level; functional data
# movement is covered by tb_reconfig_pe_array_router_lite at 4x4/8x8.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/opencode/router_lite_verilator_smoke}"
ROWS="${ROWS:-16}"
COLS="${COLS:-16}"
IDLE_CYCLES="${IDLE_CYCLES:-2}"
ELAB_ONLY="${ELAB_ONLY:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
TOP_MODULE="tb_npu_top_router_lite_elab"

VL_WARN=(
    -Wno-WIDTHEXPAND
    -Wno-WIDTHTRUNC
    -Wno-UNUSED
    -Wno-UNDRIVEN
    -Wno-UNOPTFLAT
    -Wno-PINMISSING
    -Wno-CASEINCOMPLETE
    -Wno-COMBDLY
    -Wno-INITIALDLY
    -Wno-LITENDIAN
    -Wno-DECLFILENAME
    -Wno-WIDTHCONCAT
)

RTL_FILES=(
    "$ROOT/rtl/router/router_node_lite.v"
    "$ROOT/rtl/router/router_mesh_lite.v"
    "$ROOT/rtl/router/router_local_adapter.v"
    "$ROOT/rtl/router/router_pe_mesh_lite.v"
    "$ROOT/rtl/router/router_pe_array_lite.v"
    "$ROOT/rtl/array/reconfig_pe_array.v"
    "$ROOT/rtl/buf/pingpong_buf.v"
    "$ROOT/rtl/pe/fp16_mul.v"
    "$ROOT/rtl/pe/fp32_add.v"
    "$ROOT/rtl/pe/pe_top.v"
    "$ROOT/rtl/ctrl/npu_ctrl.v"
    "$ROOT/rtl/axi/npu_axi_lite.v"
    "$ROOT/rtl/axi/npu_dma.v"
    "$ROOT/rtl/common/fifo.v"
    "$ROOT/rtl/common/axi_monitor.v"
    "$ROOT/rtl/common/op_counter.v"
    "$ROOT/rtl/power/npu_power.v"
    "$ROOT/rtl/top/npu_top.v"
)

TB_FILE="$ROOT/tb/tb_npu_top_router_lite_elab.v"
OBJ_DIR="$WORK_DIR/obj_dir_${ROWS}x${COLS}"
BIN="$OBJ_DIR/V$TOP_MODULE"

rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"

verilator --binary --timing \
    -Mdir "$OBJ_DIR" \
    --top-module "$TOP_MODULE" \
    -DROUTER_MESH_ENABLE \
    -DDUT_USE_ROUTER_MESH=1 \
    -DDUT_PHY_ROWS="$ROWS" \
    -DDUT_PHY_COLS="$COLS" \
    -DDUT_IDLE_CYCLES="$IDLE_CYCLES" \
    -DDUT_ELAB_ONLY="$ELAB_ONLY" \
    "${VL_WARN[@]}" \
    "${RTL_FILES[@]}" "$TB_FILE"

timeout "$TIMEOUT_SEC" "$BIN"
