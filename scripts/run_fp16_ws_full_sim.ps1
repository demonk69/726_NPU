# =============================================================================
# Script  : run_fp16_ws_full_sim.ps1
# Project : NPU_prj
# Desc    : Run tb_fp16_ws_full - FP16 WS full-spec verification
#           Validates ALL K individual w*a products are written back to DRAM.
#           Strategy: set N_DIM=K so dma_r_len=K*4, read all K FP32 results.
#
#   Tests (8 tests, 26 checkpoints):
#     T1: K=1  single multiply (2.0*3.0=6.0)
#     T2: K=4  all-positive    (1*1,2*1,3*1,4*1)
#     T3: K=4  mixed sign      (1*2,-1*3,2*1,-2*0.5)
#     T4: K=8  max-K burst     (0.5*[1..8])
#     T5: K=1  negative result (-3*2=-6)
#     T6: K=4  zero weights    (0*anything=0)
#     T7: K=4  fractional      (1*[0.125,0.25,0.5,1.0])
#     T8: K=2  back-to-back    (1.5*2=3, 2.5*4=10) x2
#
#   Expected: 26/26 PASS
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir  = Join-Path $ProjectRoot "rtl"
$TbDir   = Join-Path $ProjectRoot "tb"
$SimDir  = Join-Path $ProjectRoot "sim"
$WaveDir = Join-Path $SimDir "wave"
if (!(Test-Path $SimDir))  { New-Item -ItemType Directory -Path $SimDir  -Force | Out-Null }
if (!(Test-Path $WaveDir)) { New-Item -ItemType Directory -Path $WaveDir -Force | Out-Null }

$VvpOut = Join-Path $SimDir "tb_fp16_ws_full"

$Sources = @(
    (Join-Path $RtlDir "pe\fp16_mul.v"),
    (Join-Path $RtlDir "pe\fp16_add.v"),
    (Join-Path $RtlDir "pe\fp32_add.v"),
    (Join-Path $RtlDir "pe\pe_top.v"),
    (Join-Path $RtlDir "common\fifo.v"),
    (Join-Path $RtlDir "common\axi_monitor.v"),
    (Join-Path $RtlDir "common\op_counter.v"),
    (Join-Path $RtlDir "buf\pingpong_buf.v"),
    (Join-Path $RtlDir "array\pe_array.v"),
    (Join-Path $RtlDir "power\npu_power.v"),
    (Join-Path $RtlDir "ctrl\npu_ctrl.v"),
    (Join-Path $RtlDir "axi\npu_axi_lite.v"),
    (Join-Path $RtlDir "axi\npu_dma.v"),
    (Join-Path $RtlDir "top\npu_top.v"),
    (Join-Path $TbDir  "tb_fp16_ws_full.v")
)

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  NPU FP16 WS Full-Spec Simulation" -ForegroundColor Cyan
Write-Host "  tb_fp16_ws_full: 8 tests, 26 checkpoints" -ForegroundColor Cyan
Write-Host "  N_DIM=K trick: reads ALL K FP32 results from DRAM" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# Check iverilog
try {
    $ver = & iverilog -V 2>&1 | Select-Object -First 1
    Write-Host "[OK] $ver" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] iverilog not found. Install Icarus Verilog." -ForegroundColor Red
    exit 1
}

# Compile
Write-Host "`n[1/2] Compiling..." -ForegroundColor Yellow
& iverilog -g2012 -o $VvpOut $Sources 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Compilation successful." -ForegroundColor Green

# Run simulation
Write-Host "`n[2/2] Running simulation..." -ForegroundColor Yellow
$output = & vvp $VvpOut 2>&1
$output | ForEach-Object { Write-Host $_ }

# Parse result
$passLine = $output | Where-Object { $_ -match "FP16 WS FULL RESULT" }
if ($passLine) {
    if ($passLine -match "0 FAILED") {
        Write-Host "`n[RESULT] ALL PASS (26/26)" -ForegroundColor Green
    } else {
        Write-Host "`n[RESULT] FAILURES DETECTED" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n[RESULT] Could not parse result line" -ForegroundColor Yellow
}

Write-Host "`n======================================================"
Write-Host "  Add -DDUMP_VCD to iverilog for waveform output"
Write-Host "  VCD: $WaveDir\fp16_ws_full.vcd"
Write-Host "======================================================"
