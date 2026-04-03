# run_sim.ps1 -- Simulate pe_top with Icarus Verilog (Windows PowerShell)
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1

$ErrorActionPreference = "Stop"

$ProjRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$RtlPe   = "$ProjRoot\rtl\pe"
$Tb      = "$ProjRoot\tb"
$SimOut  = "$ProjRoot\sim\wave"

New-Item -ItemType Directory -Force -Path $SimOut | Out-Null

Write-Host "[INFO] Compiling..."
iverilog -g2012 `
    -I "$RtlPe" `
    -o "$SimOut\sim_pe.out" `
    "$RtlPe\fp16_mul.v" `
    "$RtlPe\fp16_add.v" `
    "$RtlPe\pe_top.v"   `
    "$Tb\tb_pe_top.v"

Write-Host "[INFO] Running simulation..."
Push-Location $SimOut
vvp sim_pe.out
Pop-Location

Write-Host "[INFO] Done. VCD: $SimOut\tb_pe_top.vcd"
Write-Host "[INFO] View with: gtkwave $SimOut\tb_pe_top.vcd"
