#!/usr/bin/env bash
# Focused direct tile-mode OS sweep for activation/weight movement controls.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/opencode/os_dataflow_modes_sweep}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
SHAPE="${SHAPE:-4x4}"
M_DIM="${M_DIM:-4}"
K_DIM="${K_DIM:-8}"
N_DIM="${N_DIM:-4}"
LANES="${LANES:-8}"

GEN_PY="$ROOT/tb/tile4/gen_multi_shape_data.py"

RTL_FILES=(
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

case "$SHAPE" in
    4x4)
        CFG_ROWS=4
        CFG_COLS=4
        ;;
    8x8)
        CFG_ROWS=8
        CFG_COLS=8
        ;;
    16x16)
        CFG_ROWS=16
        CFG_COLS=16
        ;;
    *)
        echo "[FAIL] unsupported direct OS shape: $SHAPE" >&2
        exit 1
        ;;
esac

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

CASES=(
    "a_broadcast_w_systolic 0x00"
    "a_systolic_w_systolic 0x20"
    "a_broadcast_w_broadcast 0x10"
    "a_systolic_w_broadcast 0x30"
)

total=0
passed=0

for entry in "${CASES[@]}"; do
    read -r mode_name arr_extra <<< "$entry"
    case_name="direct_${SHAPE}_os_${mode_name}_L${LANES}_M${M_DIM}_K${K_DIM}_N${N_DIM}"
    case_root="$WORK_DIR/$case_name"
    case_dir="$case_root/$case_name"
    vvp="$case_root/sim_ivl.vvp"

    total=$((total + 1))
    echo "[RUN] $case_name ARR_EXTRA=$arr_extra"

    python3 "$GEN_PY" \
        --shape "$SHAPE" --M "$M_DIM" --K "$K_DIM" --N "$N_DIM" --lanes "$LANES" \
        --flow os --arr-cfg-extra "$arr_extra" --out-dir "$case_root" --name "$case_name"

    iverilog -g2012 \
        -I "$case_dir" \
        -DDUT_PHY_ROWS="$CFG_ROWS" \
        -DDUT_PHY_COLS="$CFG_COLS" \
        ${EXTRA_IVERILOG_FLAGS:-} \
        -o "$vvp" \
        "${RTL_FILES[@]}" "$ROOT/tb/tb_npu_tile_gemm_v2.v"

    (
        cd "$case_dir"
        timeout "$TIMEOUT_SEC" vvp "$vvp" 2>&1 | grep -E 'PASS|FAIL|WARN|^\[RESULT\]|Tile GEMM Test'
    )

    passed=$((passed + 1))
done

echo "[PASS] direct OS dataflow modes sweep passed=$passed total=$total"
