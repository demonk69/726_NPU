# =============================================================================
# run_multi_shape_gemm.ps1 - Run NPU tile-mode GEMM tests for 8x8/16x16/8x32.
# =============================================================================

param(
    [string]$Shape = "8x8",
    [int]$M = 8,
    [int]$K = 4,
    [int]$N = 8,
    [string]$Name = "default",
    [switch]$DumpVcd
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RtlDir = Join-Path $ProjectRoot "rtl"
$TbDir = Join-Path $ProjectRoot "tb"
$SimDir = Join-Path $ProjectRoot "sim"
$CaseDir = Join-Path $TbDir "tile4"

if (-not (Get-Command iverilog -ErrorAction SilentlyContinue)) {
    $IcarusBin = "E:\iverilog\bin"
    if (Test-Path (Join-Path $IcarusBin "iverilog.exe")) {
        $env:Path = "$IcarusBin;$env:Path"
    }
}
if (-not (Get-Command iverilog -ErrorAction SilentlyContinue)) {
    throw "iverilog not found."
}

$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    $PythonCmd = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $PythonCmd) {
    throw "python not found."
}

Write-Host "=== NPU Multi-Shape GEMM Test: $Shape ===" -ForegroundColor Cyan

# Generate test data
$genScript = Join-Path $CaseDir "gen_multi_shape_data.py"
$caseName = "$Name" + "_${Shape}_M${M}_K${K}_N${N}"
$outDir = Join-Path $CaseDir $caseName

Write-Host "[1/3] Generating test data..." -ForegroundColor Yellow
& $PythonCmd.Source $genScript --shape $Shape --M $M --K $K --N $N --out-dir $CaseDir --name $caseName
if ($LASTEXITCODE -ne 0) {
    throw "Test data generation failed."
}

# Compile
Write-Host "[2/3] Compiling..." -ForegroundColor Yellow
$VvpFile = Join-Path $SimDir "tb_npu_multi_shape_gemm.vvp"
$srcFiles = @(
    (Join-Path $RtlDir "pe\fp16_mul.v"),
    (Join-Path $RtlDir "pe\fp16_add.v"),
    (Join-Path $RtlDir "pe\fp32_add.v"),
    (Join-Path $RtlDir "pe\pe_top.v"),
    (Join-Path $RtlDir "common\fifo.v"),
    (Join-Path $RtlDir "common\axi_monitor.v"),
    (Join-Path $RtlDir "common\op_counter.v"),
    (Join-Path $RtlDir "buf\pingpong_buf.v"),
    (Join-Path $RtlDir "buf\psum_out_buf.v"),
    (Join-Path $RtlDir "array\reconfig_pe_array.v"),
    (Join-Path $RtlDir "power\npu_power.v"),
    (Join-Path $RtlDir "ctrl\npu_ctrl.v"),
    (Join-Path $RtlDir "axi\npu_axi_lite.v"),
    (Join-Path $RtlDir "axi\npu_dma.v"),
    (Join-Path $RtlDir "top\npu_top.v"),
    (Join-Path $TbDir "tb_npu_multi_shape_gemm.v")
)

$args = @("-g2012", "-o", $VvpFile, "-s", "tb_npu_multi_shape_gemm", "-I$outDir")
if ($DumpVcd) {
    $args += "-DDUMP_VCD"
}
$args += $srcFiles

$compileOut = & iverilog @args 2>&1
if ($LASTEXITCODE -ne 0) {
    $compileOut | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    throw "iverilog compile failed."
}
Write-Host "  OK: Compiled to $VvpFile" -ForegroundColor Green

# Run simulation
Write-Host "[3/3] Running simulation..." -ForegroundColor Yellow
Push-Location (Join-Path $CaseDir $caseName)
$simOut = & vvp -N $VvpFile 2>&1
$simExit = $LASTEXITCODE
Pop-Location

$simOut | ForEach-Object { Write-Host $_ }

if (($simOut | Select-String "\[PASS\]").Count -gt 0) {
    Write-Host "=== PASS ===" -ForegroundColor Green
} else {
    Write-Host "=== FAIL ===" -ForegroundColor Red
    exit 1
}
exit $simExit
