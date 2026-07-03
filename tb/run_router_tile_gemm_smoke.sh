#!/usr/bin/env bash
# Focused tile-mode GEMM smoke for the runtime-enabled router PE-array path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/opencode/router_tile_gemm_smoke}"
SHAPE="${SHAPE:-4x4}"
M_DIM="${M_DIM:-4}"
K_DIM="${K_DIM:-8}"
N_DIM="${N_DIM:-4}"
LANES="${LANES:-8}"
FLOW="${FLOW:-os}"
CASE_NAME="${CASE_NAME:-router_${SHAPE}_${FLOW}_L${LANES}_M${M_DIM}_K${K_DIM}_N${N_DIM}}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
SIM="${SIM:-icarus}"

GEN_PY="$ROOT/tb/tile4/gen_multi_shape_data.py"
CASE_ROOT="$WORK_DIR/$CASE_NAME"
CASE_DIR="$CASE_ROOT/$CASE_NAME"
VVP="$CASE_ROOT/sim_ivl.vvp"
OBJ_DIR="$CASE_ROOT/obj_dir"
VL_BIN="$OBJ_DIR/Vtb_npu_tile_gemm_v2"

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

case "$SHAPE" in
    4x4)
        DEFAULT_PHY_ROWS=4
        DEFAULT_PHY_COLS=4
        ;;
    8x8)
        DEFAULT_PHY_ROWS=8
        DEFAULT_PHY_COLS=8
        ;;
    16x16)
        DEFAULT_PHY_ROWS=16
        DEFAULT_PHY_COLS=16
        ;;
    8x32)
        DEFAULT_PHY_ROWS=16
        DEFAULT_PHY_COLS=16
        ;;
    *)
        echo "[FAIL] unsupported router tile shape: $SHAPE" >&2
        exit 1
        ;;
esac

DUT_PHY_ROWS="${DUT_PHY_ROWS:-$DEFAULT_PHY_ROWS}"
DUT_PHY_COLS="${DUT_PHY_COLS:-$DEFAULT_PHY_COLS}"

RTL_FILES=(
    "$ROOT/rtl/router/router_node_lite.v"
    "$ROOT/rtl/router/router_mesh_lite.v"
    "$ROOT/rtl/router/router_local_adapter.v"
    "$ROOT/rtl/router/router_pe_mesh_lite.v"
    "$ROOT/rtl/router/router_pe_array_lite.v"
    "$ROOT/rtl/pe/fp16_add.v" "$ROOT/rtl/pe/fp16_mul.v" "$ROOT/rtl/pe/fp32_add.v"
    "$ROOT/rtl/pe/pe_top.v"
    "$ROOT/rtl/common/fifo.v" "$ROOT/rtl/common/op_counter.v" "$ROOT/rtl/common/axi_monitor.v"
    "$ROOT/rtl/buf/pingpong_buf.v" "$ROOT/rtl/buf/psum_out_buf.v"
    "$ROOT/rtl/array/reconfig_pe_array.v"
    "$ROOT/rtl/axi/npu_axi_lite.v" "$ROOT/rtl/axi/npu_dma.v"
    "$ROOT/rtl/ctrl/npu_ctrl.v"
    "$ROOT/rtl/power/npu_power.v"
    "$ROOT/rtl/top/npu_top.v"
)

rm -rf "$CASE_ROOT"
mkdir -p "$CASE_ROOT"

python3 "$GEN_PY" \
    --shape "$SHAPE" --M "$M_DIM" --K "$K_DIM" --N "$N_DIM" --lanes "$LANES" --flow "$FLOW" --router \
    --out-dir "$CASE_ROOT" --name "$CASE_NAME"

case "$SIM" in
    icarus)
        iverilog -g2012 \
            -I "$CASE_DIR" \
            -DROUTER_MESH_ENABLE \
            -DUSE_ROUTER_MESH_VAL=1 \
            -DDUT_PHY_ROWS="$DUT_PHY_ROWS" \
            -DDUT_PHY_COLS="$DUT_PHY_COLS" \
            ${EXTRA_IVERILOG_FLAGS:-} \
            -o "$VVP" \
            "${RTL_FILES[@]}" "$ROOT/tb/tb_npu_tile_gemm_v2.v"
        (
            cd "$CASE_DIR"
            timeout "$TIMEOUT_SEC" vvp "$VVP" 2>&1 | grep -E 'PASS|FAIL|WARN|^\[RESULT\]|Tile GEMM Test'
        )
        ;;
    verilator)
        rm -rf "$OBJ_DIR"
        mkdir -p "$OBJ_DIR"
        verilator --binary --timing \
            -I"$CASE_DIR" \
            -Mdir "$OBJ_DIR" \
            --top-module tb_npu_tile_gemm_v2 \
            -DROUTER_MESH_ENABLE \
            -DUSE_ROUTER_MESH_VAL=1 \
            -DDUT_PHY_ROWS="$DUT_PHY_ROWS" \
            -DDUT_PHY_COLS="$DUT_PHY_COLS" \
            ${EXTRA_VERILATOR_FLAGS:-} \
            "${VL_WARN[@]}" \
            "${RTL_FILES[@]}" "$ROOT/tb/tb_npu_tile_gemm_v2.v"
        (
            cd "$CASE_DIR"
            timeout "$TIMEOUT_SEC" "$VL_BIN" 2>&1 | grep -E 'PASS|FAIL|WARN|^\[RESULT\]|Tile GEMM Test'
        )
        ;;
    *)
        echo "[FAIL] unsupported SIM=$SIM" >&2
        exit 1
        ;;
esac
