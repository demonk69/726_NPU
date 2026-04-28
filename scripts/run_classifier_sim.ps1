# =============================================================================
# Script  : run_classifier_sim.ps1
# Project : NPU_prj
# Desc    : 3-Layer Tiny-FC-Net classifier inference simulation.
#           Network: FC1(16->8) -> ReLU -> FC2(8->4) -> ReLU -> FC3(4->4)
#           Uses Icarus Verilog (iverilog) + vvp.
#
# Prerequisites:
#   1. Python + numpy installed
#   2. Run this script from project root or any location
#   3. iverilog must be on PATH
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir   = Join-Path $ProjectRoot "rtl"
$TbDir    = Join-Path $ProjectRoot "tb"
$SimDir   = Join-Path $ProjectRoot "sim"
$WaveDir  = Join-Path $SimDir "wave"
$VvpOut   = Join-Path $SimDir "classifier_sim"
$HexFile  = Join-Path $TbDir "classifier_dram.hex"

# Create output dirs
if (!(Test-Path $WaveDir)) { New-Item -ItemType Directory -Path $WaveDir -Force | Out-Null }
if (!(Test-Path $SimDir))  { New-Item -ItemType Directory -Path $SimDir -Force | Out-Null }

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Tiny-FC-Net Classifier Simulation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# --- Step 0: Generate test data if needed ---
Write-Host "`n[0/4] Generating test data..." -ForegroundColor Yellow
$genScript = Join-Path $ProjectRoot "scripts\gen_classifier_data.py"
if (!(Test-Path $HexFile)) {
    Write-Host "  classifier_dram.hex not found, running gen_classifier_data.py..."
    & python $genScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Data generation failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  [OK] classifier_dram.hex exists, skipping generation." -ForegroundColor Green
    Write-Host "        (Delete it to regenerate with new random weights)" -ForegroundColor DarkGray
}

# --- Step 1: Collect sources and compile ---
Write-Host "`n[1/4] Compiling..." -ForegroundColor Yellow

$Sources = @(
    (Join-Path $RtlDir "pe\fp16_mul.v"),
    (Join-Path $RtlDir "pe\fp16_add.v"),
    (Join-Path $RtlDir "pe\fp32_add.v"),
    (Join-Path $RtlDir "pe\pe_top.v"),
    (Join-Path $RtlDir "common\fifo.v"),
    (Join-Path $RtlDir "common\axi_monitor.v"),
    (Join-Path $RtlDir "common\op_counter.v"),
    (Join-Path $RtlDir "array\reconfig_pe_array.v"),
    (Join-Path $RtlDir "power\npu_power.v"),
    (Join-Path $RtlDir "buf\pingpong_buf.v"),
    (Join-Path $RtlDir "ctrl\npu_ctrl.v"),
    (Join-Path $RtlDir "axi\npu_axi_lite.v"),
    (Join-Path $RtlDir "axi\npu_dma.v"),
    (Join-Path $RtlDir "top\npu_top.v"),
    (Join-Path $TbDir  "tb_classifier.v")
)

# Check iverilog
try {
    $ver = & iverilog -V 2>&1 | Select-Object -First 1
    Write-Host "  [OK] $ver" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] iverilog not found. Install Icarus Verilog." -ForegroundColor Red
    exit 1
}

$srcArgs = $Sources | ForEach-Object { $_ }
$defineArg = "-DDUMP_VCD"
$incArg    = "-I$TbDir"

& iverilog -o $VvpOut $defineArg $incArg $srcArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Compilation successful." -ForegroundColor Green

# --- Step 2: Run simulation ---
Write-Host "`n[2/4] Running simulation..." -ForegroundColor Yellow
Push-Location $ProjectRoot
& vvp $VvpOut 2>&1 | ForEach-Object {
    # Colorize key output
    if ($_ -match "\[PASS\]")        { Write-Host $_ -ForegroundColor Green }
    elseif ($_ -match "\[FAIL\]")     { Write-Host $_ -ForegroundColor Red }
    elseif ($_ -match "VERIFIED")     { Write-Host $_ -ForegroundColor Green }
    elseif ($_ -match "WRONG")        { Write-Host $_ -ForegroundColor Red }
    elseif ($_ -match "TIMEOUT")      { Write-Host $_ -ForegroundColor Red }
    elseif ($_ -match "NPU DONE")     { Write-Host $_ -ForegroundColor Cyan }
    elseif ($_ -match "CLASS\]")      { Write-Host $_ -ForegroundColor Magenta }
    elseif ($_ -match "STEP")         { Write-Host $_ -ForegroundColor Yellow }
    elseif ($_ -match "####")         { Write-Host $_ -ForegroundColor Cyan }
    elseif ($_ -match "=======")      { Write-Host $_ -ForegroundColor DarkGray }
    else { Write-Host $_ }
}
$vvpExit = $LASTEXITCODE
Pop-Location

# --- Step 3: Report ---
Write-Host "`n[3/4] Results:" -ForegroundColor Yellow
$VcdFile = Join-Path $WaveDir "tb_classifier.vcd"
if (Test-Path $VcdFile) {
    $vcdSize = (Get-Item $VcdFile).Length / 1KB
    Write-Host "  VCD waveform : $VcdFile ($([math]::Round($vcdSize,1)) KB)" -ForegroundColor Green
    Write-Host "  Open with    : gtkwave $VcdFile" -ForegroundColor DarkGray
} else {
    Write-Host "  No VCD file generated." -ForegroundColor Yellow
}

# --- Step 4: Summary ---
Write-Host "`n[4/4] Files:" -ForegroundColor Yellow
Write-Host "  DRAM hex     : $HexFile" -ForegroundColor DarkGray
Write-Host "  Golden ref   : $(Join-Path $TbDir 'classifier_golden.txt')" -ForegroundColor DarkGray
Write-Host "  Layout       : $(Join-Path $TbDir 'classifier_layout.txt')" -ForegroundColor DarkGray
Write-Host "  Expected .vh : $(Join-Path $TbDir 'classifier_expected.vh')" -ForegroundColor DarkGray

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Done!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
