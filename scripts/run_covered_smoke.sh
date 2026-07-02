#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/sim/covered_smoke"

rm -rf "$OUT"
mkdir -p "$OUT"
cd "$OUT"

cat > dump.v <<'EOF'
module cov_dump;
  initial begin
    $dumpfile("wave.vcd");
    $dumpvars(0, tb_npu_axi_lite_desc.dut);
  end
endmodule
EOF

iverilog -g2012 -o sim.vvp \
  "$ROOT/rtl/axi/npu_axi_lite.v" \
  "$ROOT/tb/tb_npu_axi_lite_desc.v" \
  dump.v

vvp sim.vvp | tee sim.log
grep -v 'comment Show the parameter values' wave.vcd > wave.clean.vcd

covered score \
  -t npu_axi_lite \
  -i tb_npu_axi_lite_desc.dut \
  -v "$ROOT/rtl/axi/npu_axi_lite.v" \
  -vcd wave.clean.vcd \
  -o cov.cdd

covered report -m ltc -d s cov.cdd > report.txt
sed -n '1,220p' report.txt
