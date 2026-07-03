#!/usr/bin/env bash
# Focused tile-mode GEMM sweep for the runtime-enabled router PE-array path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/opencode/router_tile_gemm_sweep}"
TIMEOUT_SEC="${TIMEOUT_SEC:-240}"
RUN_16X16_VERILATOR="${RUN_16X16_VERILATOR:-1}"
VERILATOR_TIMEOUT_SEC="${VERILATOR_TIMEOUT_SEC:-900}"
SMOKE="$ROOT/tb/run_router_tile_gemm_smoke.sh"

CASES=(
    "4x4 4 8 4"
    "4x4 3 5 2"
    "4x4 5 8 6"
    "4x4 4 12 4"
    "4x4 4 16 4"
    "4x4 4 72 4"
    "8x8 8 8 8"
    "8x8 5 5 7"
    "8x8 8 12 8"
    "8x8 8 16 8"
    "8x8 8 40 8"
)

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

total=0
passed=0

for entry in "${CASES[@]}"; do
    read -r shape m_dim k_dim n_dim <<< "$entry"
    case_name="router_${shape}_os_L8_M${m_dim}_K${k_dim}_N${n_dim}"
    total=$((total + 1))
    echo "[RUN] $case_name"
    SHAPE="$shape" \
    M_DIM="$m_dim" \
    K_DIM="$k_dim" \
    N_DIM="$n_dim" \
    LANES=8 \
    CASE_NAME="$case_name" \
    WORK_DIR="$WORK_DIR" \
    TIMEOUT_SEC="$TIMEOUT_SEC" \
        bash "$SMOKE"
    passed=$((passed + 1))
done

if [[ "$RUN_16X16_VERILATOR" == "1" ]]; then
    # 16x16 router mesh is too slow/hangs under icarus in practice; keep it on
    # Verilator so the default sweep still covers full-scale and K-split cases.
    for entry in "16x16:16:8:16" "16x16:16:16:16" "16x16:16:24:16"; do
        IFS=':' read -r shape m_dim k_dim n_dim <<< "$entry"
        case_name="router_${shape}_os_L8_M${m_dim}_K${k_dim}_N${n_dim}"
        total=$((total + 1))
        echo "[RUN] $case_name (verilator)"
        SHAPE="$shape" \
        M_DIM="$m_dim" \
        K_DIM="$k_dim" \
        N_DIM="$n_dim" \
        LANES=8 \
        CASE_NAME="$case_name" \
        WORK_DIR="$WORK_DIR" \
        TIMEOUT_SEC="$VERILATOR_TIMEOUT_SEC" \
        SIM=verilator \
            bash "$SMOKE"
        passed=$((passed + 1))
    done

    for entry in "8x32:8:8:32" "8x32:5:16:17" "8x32:8:40:32"; do
        IFS=':' read -r shape m_dim k_dim n_dim <<< "$entry"
        case_name="router_${shape}_os_L8_M${m_dim}_K${k_dim}_N${n_dim}"
        total=$((total + 1))
        echo "[RUN] $case_name (verilator)"
        SHAPE="$shape" \
        M_DIM="$m_dim" \
        K_DIM="$k_dim" \
        N_DIM="$n_dim" \
        LANES=8 \
        CASE_NAME="$case_name" \
        WORK_DIR="$WORK_DIR" \
        TIMEOUT_SEC="$VERILATOR_TIMEOUT_SEC" \
        SIM=verilator \
            bash "$SMOKE"
        passed=$((passed + 1))
    done
fi

echo "[PASS] router tile GEMM sweep passed=$passed total=$total"
