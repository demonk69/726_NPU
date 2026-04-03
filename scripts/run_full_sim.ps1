# =============================================================================
# Script  : run_full_sim.ps1
# Project : NPU_prj
# Desc    : NPU full-system simulation with bandwidth & operation statistics.
#           Uses Icarus Verilog (iverilog) + vvp.
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir   = Join-Path $ProjectRoot "rtl"
$TbDir    = Join-Path $ProjectRoot "tb"
$SimDir   = Join-Path $ProjectRoot "sim"
$WaveDir  = Join-Path $SimDir "wave"
$VvpOut   = Join-Path $SimDir "npu_sim"

# Create output dirs
if (!(Test-Path $WaveDir)) { New-Item -ItemType Directory -Path $WaveDir -Force | Out-Null }
if (!(Test-Path $SimDir))  { New-Item -ItemType Directory -Path $SimDir -Force | Out-Null }

# Collect all RTL sources
$Sources = @(
    (Join-Path $RtlDir "pe\fp16_mul.v"),
    (Join-Path $RtlDir "pe\fp16_add.v"),
    (Join-Path $RtlDir "pe\pe_top.v"),
    (Join-Path $RtlDir "common\fifo.v"),
    (Join-Path $RtlDir "common\axi_monitor.v"),
    (Join-Path $RtlDir "common\op_counter.v"),
    (Join-Path $RtlDir "array\pe_array.v"),
    (Join-Path $RtlDir "power\npu_power.v"),
    (Join-Path $RtlDir "ctrl\npu_ctrl.v"),
    (Join-Path $RtlDir "axi\npu_axi_lite.v"),
    (Join-Path $RtlDir "axi\npu_dma.v"),
    (Join-Path $RtlDir "top\npu_top.v"),
    (Join-Path $TbDir  "tb_npu_top.v")
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NPU Full System Simulation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check iverilog
try {
    $ver = & iverilog -V 2>&1 | Select-Object -First 1
    Write-Host "[OK] $ver" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] iverilog not found. Please install Icarus Verilog." -ForegroundColor Red
    exit 1
}

# Step 1: Compile
Write-Host "`n[1/3] Compiling..." -ForegroundColor Yellow
$srcArgs = $Sources | ForEach-Object { $_ }
$defineArg = "-DDUMP_VCD"
$includeArg = "-I$RtlDir"

& iverilog -o $VvpOut $defineArg $includeArg $srcArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Compilation successful." -ForegroundColor Green

# Step 2: Run simulation
Write-Host "`n[2/3] Running simulation..." -ForegroundColor Yellow
Push-Location $WaveDir
& vvp $VvpOut 2>&1
$vvpExit = $LASTEXITCODE
Pop-Location

if ($vvpExit -ne 0) {
    Write-Host "[WARN] Simulation exited with code $vvpExit" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Simulation completed." -ForegroundColor Green
}

# Step 3: Report
$VcdFile = Join-Path $WaveDir "tb_npu_top.vcd"
if (Test-Path $VcdFile) {
    $vcdSize = (Get-Item $VcdFile).Length / 1KB
    Write-Host "`n[3/3] Results:" -ForegroundColor Yellow
    Write-Host "  VCD waveform : $VcdFile ($([math]::Round($vcdSize,1)) KB)" -ForegroundColor Green
    Write-Host "  Open with    : gtkwave $VcdFile" -ForegroundColor DarkGray
} else {
    Write-Host "`n[3/3] No VCD file generated." -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Done!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
