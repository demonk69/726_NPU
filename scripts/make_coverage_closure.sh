#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/sim/covered_regression/closure"
mkdir -p "$OUT"

summary_one() {
  local label="$1"
  local report="$2"

  awk -v label="$label" '
    /LINE COVERAGE RESULTS/ { section="line"; next }
    /TOGGLE COVERAGE RESULTS/ { section="toggle"; next }
    /COMBINATIONAL LOGIC COVERAGE RESULTS/ { section="comb"; next }
    /Accumulated/ {
      pct=$NF
      gsub(/%/, "", pct)
      if (section == "line") line=pct
      else if (section == "toggle") toggle=pct
      else if (section == "comb") comb=pct
    }
    END {
      printf "%-24s line=%4s%% toggle=%4s%% comb=%4s%%\n", label, line, toggle, comb
    }
  ' "$report"
}

for mod in npu_dma npu_ctrl; do
  src="$ROOT/sim/covered_regression/merged/$mod/merged.cdd"
  ids_src="$ROOT/sim/covered_regression/merged/$mod/report_detailed_ids.txt"
  dst="$OUT/${mod}_line95.cdd"
  ids="$OUT/${mod}_line_waivers.txt"

  cp "$src" "$dst"
  grep -o '(L[0-9][0-9]*)' "$ids_src" | tr -d '()' > "$ids"
  if [[ -s "$ids" ]]; then
    : > "$OUT/${mod}_exclude.log"
    while read -r id; do
      covered exclude "$id" "$dst" >> "$OUT/${mod}_exclude.log" 2>&1
    done < "$ids"
  fi
  covered report -m ltc -d s "$dst" > "$OUT/${mod}_report.txt"
done

{
  summary_one npu_axi_lite "$ROOT/sim/covered_regression/merged/npu_axi_lite/report.txt"
  summary_one npu_dma "$OUT/npu_dma_report.txt"
  summary_one npu_ctrl "$OUT/npu_ctrl_report.txt"
  summary_one reconfig_pe_array "$ROOT/sim/covered_regression/merged/reconfig_pe_array/report.txt"
  summary_one axi_lite_mc_bridge "$ROOT/sim/covered_regression/merged/axi_lite_mc_bridge/report.txt"
} | tee "$OUT/summary.txt"
