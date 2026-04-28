# =============================================================================
# run_npu_top_sanity.ps1  -  NPU Top-Level Sanity Check (Smoke Test)
#
# Tests a toy INT8 OS-mode matrix multiply (M=4, N=1, K=4) against
# known-good golden results, exercising the full DMA → PPBuf → PE → WB path.
#
# Usage (from project root):
#   .\scripts\run_npu_top_sanity.ps1          # normal run
#   .\scripts\run_npu_top_sanity.ps1 -Dump    # also save VCD waveform
#
# Prerequisites:
#   - Icarus Verilog (iverilog / vvp) on PATH
# =============================================================================

param(
    [switch]$Dump   # pass -Dump to enable VCD waveform output
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "D:\NPU_prj"
$RtlDir  = Join-Path $ProjectRoot "rtl"
$SimDir  = Join-Path $ProjectRoot "sim"

if (-not (Test-Path $SimDir)) {
    New-Item -ItemType Directory -Path $SimDir | Out-Null
}

Write-Host "=== NPU Top Sanity Check ===" -ForegroundColor Cyan
Write-Host ""

# ---- Step 1: Compile ----
Write-Host "[1/2] Compiling Verilog..." -ForegroundColor Yellow

$srcFiles = @(
    # PE arithmetic
    (Join-Path $RtlDir "pe\fp16_add.v")
    (Join-Path $RtlDir "pe\fp16_mul.v")
    (Join-Path $RtlDir "pe\fp32_add.v")
    (Join-Path $RtlDir "pe\pe_top.v")
    (Join-Path $RtlDir "array\reconfig_pe_array.v")
    # Buffers / FIFO
    (Join-Path $RtlDir "common\fifo.v")
    (Join-Path $RtlDir "buf\pingpong_buf.v")
    # AXI / DMA / Ctrl
    (Join-Path $RtlDir "axi\npu_axi_lite.v")
    (Join-Path $RtlDir "axi\npu_dma.v")
    (Join-Path $RtlDir "ctrl\npu_ctrl.v")
    # Power (stub)
    (Join-Path $RtlDir "power\npu_power.v")
    # Top
    (Join-Path $RtlDir "top\npu_top.v")
    # Testbench
    (Join-Path $SimDir "tb_npu_top.v")
)

$vvpFile = Join-Path $SimDir "tb_npu_top.vvp"
$compArgs = @("-g2012", "-o", $vvpFile, "-s", "tb_npu_top")

foreach ($f in $srcFiles) {
    if (Test-Path $f) {
        $compArgs += $f
    } else {
        Write-Host "  WARNING: Source file not found: $f" -ForegroundColor DarkYellow
    }
}

& iverilog @compArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Compilation failed." -ForegroundColor Red
    exit 1
}
Write-Host "  OK: Compiled -> $vvpFile" -ForegroundColor Green

# ---- Step 2: Run simulation ----
Write-Host "[2/2] Running simulation..." -ForegroundColor Yellow

Push-Location $SimDir
if ($Dump) {
    Write-Host "  VCD dump enabled (+DUMP flag)" -ForegroundColor DarkGray
    & vvp -N $vvpFile "+DUMP"
} else {
    & vvp -N $vvpFile
}
$simExit = $LASTEXITCODE
Pop-Location

Write-Host ""
if ($simExit -eq 0) {
    Write-Host "=== Simulation completed (exit 0) ===" -ForegroundColor Green
} else {
    Write-Host "=== Simulation completed (exit $simExit) ===" -ForegroundColor Yellow
}

if ($Dump) {
    $vcdFile = Join-Path $SimDir "tb_npu_top.vcd"
    if (Test-Path $vcdFile) {
        Write-Host "  Waveform: $vcdFile" -ForegroundColor DarkGray
        Write-Host "  View with: gtkwave $vcdFile" -ForegroundColor DarkGray
    }
}

exit $simExit
