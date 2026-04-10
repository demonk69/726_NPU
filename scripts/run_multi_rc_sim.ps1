# =============================================================================
# Script  : run_multi_rc_sim.ps1
# Project : NPU_prj
# Desc    : Run tb_multi_rc_comprehensive (ROWS=2, COLS=2 PE array test)
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir  = Join-Path $ProjectRoot "rtl"
$TbDir   = Join-Path $ProjectRoot "tb"
$SimDir  = Join-Path $ProjectRoot "sim"
if (!(Test-Path $SimDir)) { New-Item -ItemType Directory -Path $SimDir -Force | Out-Null }

$VvpOut  = Join-Path $SimDir "tb_multi_rc"

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
    (Join-Path $TbDir  "tb_multi_rc_comprehensive.v")
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NPU Multi-Row/Col Comprehensive Test" -ForegroundColor Cyan
Write-Host "  ROWS=2, COLS=2" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check iverilog
try {
    $ver = & iverilog -V 2>&1 | Select-Object -First 1
    Write-Host "[OK] $ver" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] iverilog not found." -ForegroundColor Red; exit 1
}

# Compile
Write-Host "`n[1/2] Compiling..." -ForegroundColor Yellow
& iverilog -o $VvpOut -I $RtlDir -DDUMP_VCD $Sources 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red; exit 1
}
Write-Host "[OK] Compilation successful." -ForegroundColor Green

# Run
Write-Host "`n[2/2] Running simulation..." -ForegroundColor Yellow
$output = & vvp $VvpOut 2>&1
$vvpExit = $LASTEXITCODE
$output | ForEach-Object { Write-Host $_ }

if ($vvpExit -ne 0) {
    Write-Host "[WARN] vvp exited with code $vvpExit" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Simulation completed." -ForegroundColor Green
}

Write-Host "`n========================================"
Write-Host "  Done!"
Write-Host "========================================"
