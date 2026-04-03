#!/bin/bash
# =============================================================================
# run_sim.sh -- Simulate pe_top with Icarus Verilog
# Usage: bash scripts/run_sim.sh
# =============================================================================

set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RTL_PE="$PROJ_ROOT/rtl/pe"
TB="$PROJ_ROOT/tb"
SIM_OUT="$PROJ_ROOT/sim/wave"

mkdir -p "$SIM_OUT"

echo "[INFO] Compiling..."
iverilog -g2012 \
    -I "$RTL_PE" \
    -o "$SIM_OUT/sim_pe.out" \
    "$RTL_PE/fp16_mul.v" \
    "$RTL_PE/fp16_add.v" \
    "$RTL_PE/fp32_add.v" \
    "$RTL_PE/pe_top.v"   \
    "$TB/tb_pe_top.v"

echo "[INFO] Running simulation..."
cd "$SIM_OUT"
vvp sim_pe.out

echo "[INFO] Done. VCD: $SIM_OUT/tb_pe_top.vcd"
echo "[INFO] Open with: gtkwave $SIM_OUT/tb_pe_top.vcd"
