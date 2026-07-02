# =============================================================================
# run_pth_multilayer_soc.ps1 - Run a 3-layer .pth-converted Conv/ReLU SoC smoke.
# =============================================================================

param(
    [switch]$DumpVcd
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RtlDir = Join-Path $ProjectRoot "rtl"
$TbDir = Join-Path $ProjectRoot "tb"
$SimDir = Join-Path $ProjectRoot "sim"
$Picorv32Dir = Join-Path $ProjectRoot "picorv32_ref"
$CaseDir = Join-Path $SimDir "pth_multilayer_conv"

if (-not (Get-Command iverilog -ErrorAction SilentlyContinue)) {
    $IcarusBin = "E:\iverilog\bin"
    if (Test-Path (Join-Path $IcarusBin "iverilog.exe")) {
        $env:Path = "$IcarusBin;$env:Path"
    }
}
if (-not (Get-Command iverilog -ErrorAction SilentlyContinue)) {
    throw "iverilog not found. Add Icarus Verilog to PATH first."
}
if (-not (Get-Command vvp -ErrorAction SilentlyContinue)) {
    throw "vvp not found. Add Icarus Verilog to PATH first."
}

$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    $PythonCmd = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $PythonCmd) {
    throw "python not found. Add Python/PyTorch environment to PATH first."
}

if (-not (Test-Path $SimDir)) {
    New-Item -ItemType Directory -Path $SimDir | Out-Null
}

Write-Host "=== PTH Multilayer Conv SoC Simulation ===" -ForegroundColor Cyan
Write-Host "[1/3] Generating 3-layer .pth, NPU assets, DRAM image, and firmware..." -ForegroundColor Yellow
& $PythonCmd.Source (Join-Path $ProjectRoot "tools\pth\gen_tiny_multilayer_soc_case.py") --out-dir $CaseDir
if ($LASTEXITCODE -ne 0) {
    throw "multilayer pth SoC case generation failed."
}

Copy-Item (Join-Path $Picorv32Dir "picorv32.v") (Join-Path $SimDir "picorv32.v") -Force

Write-Host "[2/3] Compiling Verilog..." -ForegroundColor Yellow
$VvpFile = Join-Path $SimDir "soc_pth_multilayer_conv.vvp"
$srcFiles = @(
    (Join-Path $SimDir "picorv32.v"),
    (Join-Path $RtlDir "soc\soc_mem.v"),
    (Join-Path $RtlDir "soc\dram_model.v"),
    (Join-Path $RtlDir "soc\axi_lite_bridge.v"),
    (Join-Path $RtlDir "soc\soc_top.v"),
    (Join-Path $RtlDir "axi\npu_axi_lite.v"),
    (Join-Path $RtlDir "ctrl\npu_ctrl.v"),
    (Join-Path $RtlDir "axi\npu_dma.v"),
    (Join-Path $RtlDir "buf\pingpong_buf.v"),
    (Join-Path $RtlDir "buf\psum_out_buf.v"),
    (Join-Path $RtlDir "common\fifo.v"),
    (Join-Path $RtlDir "common\axi_monitor.v"),
    (Join-Path $RtlDir "common\op_counter.v"),
    (Join-Path $RtlDir "pe\pe_top.v"),
    (Join-Path $RtlDir "pe\fp16_add.v"),
    (Join-Path $RtlDir "pe\fp16_mul.v"),
    (Join-Path $RtlDir "pe\fp32_add.v"),
    (Join-Path $RtlDir "array\reconfig_pe_array.v"),
    (Join-Path $RtlDir "power\npu_power.v"),
    (Join-Path $RtlDir "top\npu_top.v"),
    (Join-Path $TbDir "tb_soc_pth_multilayer_conv.v")
)

$args = @("-g2012", "-o", $VvpFile, "-s", "tb_soc_pth_multilayer_conv", "-I$CaseDir")
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

Write-Host "[3/3] Running simulation..." -ForegroundColor Yellow
Push-Location $SimDir
$simOut = & vvp -N $VvpFile 2>&1
$simExit = $LASTEXITCODE
Pop-Location

$simOut | ForEach-Object { Write-Host $_ }

if (($simOut | Select-String "\[FAIL\]|\[TIMEOUT\]|FATAL|ERROR:").Count -gt 0) {
    exit 1
}
if (($simOut | Select-String "\[PASS\] PTH multilayer Conv SoC test PASSED").Count -eq 0) {
    exit 1
}

Write-Host "=== Done ===" -ForegroundColor Cyan
exit $simExit
