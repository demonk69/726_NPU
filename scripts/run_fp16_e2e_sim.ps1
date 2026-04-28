# =============================================================================
# Script  : run_fp16_e2e_sim.ps1
# Project : NPU_prj
# Desc    : Run tb_fp16_e2e - FP16 end-to-end pipeline verification
#           Tests: OS/WS x K=4/8 x positive/negative/zero/precision
#           Expected: 8/8 PASS
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir  = Join-Path $ProjectRoot "rtl"
$TbDir   = Join-Path $ProjectRoot "tb"
$SimDir  = Join-Path $ProjectRoot "sim"
$WaveDir = Join-Path $SimDir "wave"
if (!(Test-Path $SimDir))  { New-Item -ItemType Directory -Path $SimDir  -Force | Out-Null }
if (!(Test-Path $WaveDir)) { New-Item -ItemType Directory -Path $WaveDir -Force | Out-Null }

$VvpOut = Join-Path $SimDir "tb_fp16_e2e"

$Sources = @(
    (Join-Path $RtlDir "pe\fp16_mul.v"),
    (Join-Path $RtlDir "pe\fp16_add.v"),
    (Join-Path $RtlDir "pe\fp32_add.v"),
    (Join-Path $RtlDir "pe\pe_top.v"),
    (Join-Path $RtlDir "common\fifo.v"),
    (Join-Path $RtlDir "common\axi_monitor.v"),
    (Join-Path $RtlDir "common\op_counter.v"),
    (Join-Path $RtlDir "buf\pingpong_buf.v"),
    (Join-Path $RtlDir "array\reconfig_pe_array.v"),
    (Join-Path $RtlDir "power\npu_power.v"),
    (Join-Path $RtlDir "ctrl\npu_ctrl.v"),
    (Join-Path $RtlDir "axi\npu_axi_lite.v"),
    (Join-Path $RtlDir "axi\npu_dma.v"),
    (Join-Path $RtlDir "top\npu_top.v"),
    (Join-Path $TbDir  "tb_fp16_e2e.v")
)

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  NPU FP16 End-to-End Simulation" -ForegroundColor Cyan
Write-Host "  tb_fp16_e2e: 7 tests, 8 checkpoints" -ForegroundColor Cyan
Write-Host "  OS/WS x K=4/8 x pos/neg/zero/precision/b2b" -ForegroundColor Cyan
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
& iverilog -g2012 -o $VvpOut -DDUMP_VCD $Sources 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Compilation successful." -ForegroundColor Green

# Run simulation
Write-Host "`n[2/2] Running simulation..." -ForegroundColor Yellow
$output = & vvp $VvpOut 2>&1
$vvpExit = $LASTEXITCODE
$output | ForEach-Object { Write-Host $_ }

# Parse PASS/FAIL
$passLine = $output | Where-Object { $_ -match "FP16 E2E RESULT" }
if ($passLine) {
    if ($passLine -match "0 FAILED") {
        Write-Host "`n[RESULT] ALL PASS" -ForegroundColor Green
    } else {
        Write-Host "`n[RESULT] FAILURES DETECTED" -ForegroundColor Red
    }
}

Write-Host "`n======================================================"
Write-Host "  VCD: $WaveDir\fp16_e2e.vcd"
Write-Host "  Open with: gtkwave $WaveDir\fp16_e2e.vcd"
Write-Host "======================================================"
