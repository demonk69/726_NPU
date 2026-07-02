#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/sim/covered_regression"

rm -rf "$OUT"
mkdir -p "$OUT"

pass_count=0
fail_count=0
xfail_count=0

TOP_RTL=(
  "rtl/pe/fp16_add.v"
  "rtl/pe/fp16_mul.v"
  "rtl/pe/fp32_add.v"
  "rtl/pe/pe_top.v"
  "rtl/array/reconfig_pe_array.v"
  "rtl/buf/pingpong_buf.v"
  "rtl/buf/psum_out_buf.v"
  "rtl/common/fifo.v"
  "rtl/common/axi_monitor.v"
  "rtl/common/op_counter.v"
  "rtl/power/npu_power.v"
  "rtl/axi/npu_dma.v"
  "rtl/axi/npu_axi_lite.v"
  "rtl/ctrl/npu_ctrl.v"
  "rtl/top/npu_top.v"
)

run_cov() {
  local name="$1"
  local top="$2"
  local inst="$3"
  local rtl_csv="$4"
  local tb="$5"
  shift 5
  local extra=("$@")

  local dir="$OUT/$name"
  mkdir -p "$dir"

  {
    echo "=== $name ==="
    echo "top  : $top"
    echo "inst : $inst"

    cat > "$dir/dump.v" <<EOF
module cov_dump_${name};
  initial begin
    \$dumpfile("wave.vcd");
    \$dumpvars(0, ${inst});
  end
endmodule
EOF

    IFS=',' read -r -a rtl_files <<< "$rtl_csv"
    local src=()
    for f in "${rtl_files[@]}"; do
      src+=("$ROOT/$f")
    done
    for f in "${extra[@]}"; do
      src+=("$ROOT/$f")
    done
    src+=("$ROOT/$tb" "$dir/dump.v")

    (cd "$dir" && iverilog -g2012 -o sim.vvp "${src[@]}")
    (cd "$dir" && vvp sim.vvp | tee sim.log)

    if grep -q '\[FAIL\]\|TIMEOUT\|timeout' "$dir/sim.log"; then
      echo "[COV_FAIL] simulation reported failure/timeout"
      return 1
    fi
    if ! grep -q '\[PASS\]\|ALL .* PASSED' "$dir/sim.log"; then
      echo "[COV_FAIL] simulation did not report PASS"
      return 1
    fi

    grep -v 'comment Show the parameter values' "$dir/wave.vcd" > "$dir/wave.clean.vcd"

    local cov_args=(score -t "$top" -i "$inst")
    local cov_files=("${rtl_files[@]}" "${extra[@]}")
    for f in "${cov_files[@]}"; do
      cov_args+=(-v "$ROOT/$f")
    done
    cov_args+=(-vcd "$dir/wave.clean.vcd" -o "$dir/cov.cdd")

    (cd "$dir" && covered "${cov_args[@]}")
    (cd "$dir" && covered report -m ltc -d s cov.cdd > report.txt)
    sed -n '/LINE COVERAGE RESULTS/,+18p' "$dir/report.txt"
    sed -n '/COMBINATIONAL LOGIC COVERAGE RESULTS/,+18p' "$dir/report.txt"
  } > "$dir/run.log" 2>&1

  cat "$dir/run.log"
  if grep -q '^ERROR!\|Segmentation fault\|CDD file was found to be empty\|Attempting to generate report on non-scored design' "$dir/run.log"; then
    return 1
  fi
  test -s "$dir/cov.cdd"
  test -s "$dir/report.txt"
}

try_cov() {
  local name="$1"
  if run_cov "$@"; then
    pass_count=$((pass_count + 1))
    echo "[COV_PASS] $name"
  else
    fail_count=$((fail_count + 1))
    echo "[COV_FAIL] $name"
  fi
}

run_shape_cov() {
  local name="$1"
  local shape="$2"
  local m_dim="$3"
  local k_dim="$4"
  local n_dim="$5"
  local lanes="$6"

  local dir="$OUT/$name"
  local data_dir="$dir/data"
  local case_dir="$data_dir/$name"
  mkdir -p "$dir"

  {
    echo "=== $name ==="
    echo "top  : npu_top"
    echo "inst : tb_npu_tile_gemm_v2.u_npu"

    python3 "$ROOT/tb/tile4/gen_multi_shape_data.py" \
      --shape "$shape" \
      --M "$m_dim" \
      --K "$k_dim" \
      --N "$n_dim" \
      --lanes "$lanes" \
      --out-dir "$data_dir" \
      --name "$name" >/dev/null

    cat > "$dir/dump.v" <<EOF
module cov_dump_${name};
  initial begin
    \$dumpfile("$dir/wave.vcd");
    \$dumpvars(0, tb_npu_tile_gemm_v2.u_npu);
  end
endmodule
EOF

    local src=()
    for f in "${TOP_RTL[@]}"; do
      src+=("$ROOT/$f")
    done
    src+=("$ROOT/tb/tb_npu_tile_gemm_v2.v" "$dir/dump.v")

    iverilog -g2012 -I "$case_dir" -o "$dir/sim.vvp" "${src[@]}"
    (cd "$ROOT" && timeout 90 vvp "$dir/sim.vvp" | tee "$dir/sim.log")

    if grep -q '\[FAIL\]\|TIMEOUT\|timeout' "$dir/sim.log"; then
      echo "[COV_FAIL] simulation reported failure/timeout"
      return 1
    fi
    if ! grep -q '\[PASS\]\|ALL .* PASSED' "$dir/sim.log"; then
      echo "[COV_FAIL] simulation did not report PASS"
      return 1
    fi

    grep -v 'comment Show the parameter values' "$dir/wave.vcd" > "$dir/wave.clean.vcd"

    local cov_args=(score -t npu_top -i tb_npu_tile_gemm_v2.u_npu)
    for f in "${TOP_RTL[@]}"; do
      cov_args+=(-v "$ROOT/$f")
    done
    cov_args+=(-vcd "$dir/wave.clean.vcd" -o "$dir/cov.cdd")

    (cd "$dir" && covered "${cov_args[@]}")
    (cd "$dir" && covered report -m ltc -d s cov.cdd > report.txt)
    sed -n '/LINE COVERAGE RESULTS/,+25p' "$dir/report.txt"
    sed -n '/COMBINATIONAL LOGIC COVERAGE RESULTS/,+25p' "$dir/report.txt"
  } > "$dir/run.log" 2>&1

  cat "$dir/run.log"
  if grep -q '^ERROR!\|Segmentation fault\|CDD file was found to be empty\|Attempting to generate report on non-scored design' "$dir/run.log"; then
    return 1
  fi
  test -s "$dir/cov.cdd"
  test -s "$dir/report.txt"
}

try_shape_cov() {
  local name="$1"
  if run_shape_cov "$@"; then
    pass_count=$((pass_count + 1))
    echo "[COV_PASS] $name"
  else
    fail_count=$((fail_count + 1))
    echo "[COV_FAIL] $name"
  fi
}

xfail_shape_cov() {
  local name="$1"
  if run_shape_cov "$@"; then
    pass_count=$((pass_count + 1))
    echo "[COV_PASS] $name"
  else
    xfail_count=$((xfail_count + 1))
    echo "[COV_XFAIL] $name"
  fi
}

xfail_cov() {
  local name="$1"
  if run_cov "$@"; then
    pass_count=$((pass_count + 1))
    echo "[COV_PASS] $name"
  else
    xfail_count=$((xfail_count + 1))
    echo "[COV_XFAIL] $name"
  fi
}

report_summary() {
  local label="$1"
  local report="$2"

  awk -v label="$label" '
    /LINE COVERAGE RESULTS/ { section="line"; next }
    /TOGGLE COVERAGE RESULTS/ { section="toggle"; next }
    /COMBINATIONAL LOGIC COVERAGE RESULTS/ { section="comb"; next }
    /FSM COVERAGE RESULTS/ { section="fsm"; next }
    /Accumulated/ {
      pct=$NF
      gsub(/%/, "", pct)
      if (section == "line") line=pct
      else if (section == "toggle") toggle=pct
      else if (section == "comb") comb=pct
    }
    END {
      if (line == "") line="NA"
      if (toggle == "") toggle="NA"
      if (comb == "") comb="NA"
      printf "%-24s line=%4s%% toggle=%4s%% comb=%4s%%\n", label, line, toggle, comb
    }
  ' "$report"
}

merge_group() {
  local label="$1"
  local top="$2"
  shift 2
  local cases=("$@")
  local dir="$OUT/merged/$label"
  mkdir -p "$dir"

  local cdds=()
  for case_name in "${cases[@]}"; do
    if [[ -s "$OUT/$case_name/cov.cdd" ]]; then
      cdds+=("$OUT/$case_name/cov.cdd")
    fi
  done

  if [[ "${#cdds[@]}" -eq 0 ]]; then
    echo "[MERGE_SKIP] $label: no CDD files"
    return 1
  fi

  if [[ "${#cdds[@]}" -eq 1 ]]; then
    cp "${cdds[0]}" "$dir/merged.cdd"
  else
    covered merge -o "$dir/merged.cdd" "${cdds[@]}" > "$dir/merge.log" 2>&1
  fi

  covered report -m ltc -d s "$dir/merged.cdd" > "$dir/report.txt" 2> "$dir/report.err"
  report_summary "$label" "$dir/report.txt" | tee "$dir/summary.txt"
}

try_cov axi_lite_desc \
  npu_axi_lite tb_npu_axi_lite_desc.dut \
  "rtl/axi/npu_axi_lite.v" \
  "tb/tb_npu_axi_lite_desc.v"

try_cov axi_lite_regs_perf \
  npu_axi_lite tb_npu_axi_lite_regs_perf.dut \
  "rtl/axi/npu_axi_lite.v" \
  "tb/tb_npu_axi_lite_regs_perf.v"

try_cov dma_read_burst \
  npu_dma tb_dma_read_burst.dut \
  "rtl/common/fifo.v,rtl/axi/npu_dma.v" \
  "tb/tb_dma_read_burst.v"

try_cov dma_write_burst \
  npu_dma tb_dma_write_burst.dut \
  "rtl/common/fifo.v,rtl/axi/npu_dma.v" \
  "tb/tb_dma_write_burst.v"

try_cov dma_burst \
  npu_dma tb_dma_burst.dut \
  "rtl/common/fifo.v,rtl/axi/npu_dma.v" \
  "tb/tb_dma_burst.v"

try_cov dma_perf \
  npu_dma tb_dma_perf.dut \
  "rtl/common/fifo.v,rtl/axi/npu_dma.v" \
  "tb/tb_dma_perf.v"

try_cov dma_adv_paths \
  npu_dma tb_dma_adv_paths.dut \
  "rtl/common/fifo.v,rtl/axi/npu_dma.v" \
  "tb/tb_dma_adv_paths.v"

try_cov ctrl_error_status \
  npu_ctrl tb_npu_ctrl_error_status.dut \
  "rtl/ctrl/npu_ctrl.v" \
  "tb/tb_npu_ctrl_error_status.v"

try_cov ctrl_tile \
  npu_ctrl tb_npu_ctrl_tile.dut \
  "rtl/ctrl/npu_ctrl.v" \
  "tb/tb_npu_ctrl_tile.v"

try_cov ctrl_ksplit \
  npu_ctrl tb_npu_ctrl_ksplit.dut \
  "rtl/ctrl/npu_ctrl.v" \
  "tb/tb_npu_ctrl_ksplit.v"

try_cov ctrl_dataflow \
  npu_ctrl tb_npu_ctrl_dataflow_modes.dut \
  "rtl/ctrl/npu_ctrl.v" \
  "tb/tb_npu_ctrl_dataflow_modes.v"

try_cov ctrl_desc_flow \
  npu_ctrl tb_npu_ctrl_desc_flow.dut \
  "rtl/ctrl/npu_ctrl.v" \
  "tb/tb_npu_ctrl_desc_flow.v"

try_cov ctrl_extra_paths \
  npu_ctrl tb_npu_ctrl_extra_paths.dut \
  "rtl/ctrl/npu_ctrl.v" \
  "tb/tb_npu_ctrl_extra_paths.v"

try_cov reconfig_acc_init \
  reconfig_pe_array tb_reconfig_pe_acc_init.dut \
  "rtl/pe/fp16_mul.v,rtl/pe/fp32_add.v,rtl/pe/pe_top.v,rtl/array/reconfig_pe_array.v" \
  "tb/tb_reconfig_pe_acc_init.v"

try_cov reconfig_8x32 \
  reconfig_pe_array tb_reconfig_pe_8x32.dut \
  "rtl/pe/fp16_mul.v,rtl/pe/fp32_add.v,rtl/pe/pe_top.v,rtl/array/reconfig_pe_array.v" \
  "tb/tb_reconfig_pe_8x32.v"

try_cov axi_lite_mc_bridge \
  axi_lite_mc_bridge tb_axi_lite_mc_bridge.dut \
  "rtl/soc/axi_lite_mc_bridge.v" \
  "tb/tb_axi_lite_mc_bridge.v"

xfail_cov dram_multi_port \
  dram_multi_port tb_dram_multi_port.dut \
  "rtl/soc/dram_multi_port.v" \
  "tb/tb_dram_multi_port.v"

echo "=== COVERED REGRESSION SUMMARY ==="
echo "PASS: $pass_count"
echo "FAIL: $fail_count"
echo "XFAIL: $xfail_count"

echo "=== MERGED COVERAGE SUMMARY ==="
mkdir -p "$OUT/merged"
merge_group npu_axi_lite npu_axi_lite axi_lite_desc axi_lite_regs_perf
merge_group npu_dma npu_dma dma_read_burst dma_write_burst dma_burst dma_perf dma_adv_paths
merge_group npu_ctrl npu_ctrl ctrl_error_status ctrl_tile ctrl_ksplit ctrl_dataflow ctrl_desc_flow ctrl_extra_paths
merge_group reconfig_pe_array reconfig_pe_array reconfig_acc_init reconfig_8x32
merge_group axi_lite_mc_bridge axi_lite_mc_bridge axi_lite_mc_bridge

if [[ "$fail_count" -ne 0 ]]; then
  exit 1
fi
