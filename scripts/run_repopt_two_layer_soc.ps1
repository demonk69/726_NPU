# =============================================================================
# run_repopt_two_layer_soc.ps1 - Run a two-layer RepOpt SoC RTL window.
#
# stage1_0 full layer -> CPU repack -> stage1_1 selected tile window
# =============================================================================

param(
    [int]$Index = 0,
    [int]$MBase2 = 0,
    [int]$NBase2 = 0,
    [int]$MTiles2 = 2,
    [int]$NTiles2 = 2,
    [switch]$WithPool,
    [int]$PoolRowBase = 0,
    [int]$PoolRows = 1,
    [int]$PoolColBase = 0,
    [int]$PoolCols = 4,
    [switch]$DumpVcd
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RtlDir = Join-Path $ProjectRoot "rtl"
$TbDir = Join-Path $ProjectRoot "tb"
$SimDir = Join-Path $ProjectRoot "sim"
$Picorv32Dir = Join-Path $ProjectRoot "picorv32_ref"
$CaseDir = Join-Path $SimDir "repopt_two_layer_soc"

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

Write-Host "=== RepOpt Two-Layer SoC Simulation ===" -ForegroundColor Cyan
Write-Host "[1/3] Generating two-layer DRAM image and CPU firmware..." -ForegroundColor Yellow
$genArgs = @(
    (Join-Path $ProjectRoot "tools\pth\gen_repopt_two_layer_soc_case.py"),
    "--out-dir", $CaseDir,
    "--index", $Index,
    "--m-base2", $MBase2,
    "--n-base2", $NBase2,
    "--m-tiles2", $MTiles2,
    "--n-tiles2", $NTiles2
)
if ($WithPool) {
    $genArgs += @("--with-pool", "--pool-row-base", $PoolRowBase, "--pool-rows", $PoolRows, "--pool-col-base", $PoolColBase, "--pool-cols", $PoolCols)
}
& $PythonCmd.Source @genArgs
if ($LASTEXITCODE -ne 0) {
    throw "RepOpt two-layer SoC case generation failed."
}

Copy-Item (Join-Path $Picorv32Dir "picorv32.v") (Join-Path $SimDir "picorv32.v") -Force

Write-Host "[2/3] Compiling Verilog..." -ForegroundColor Yellow
$VvpFile = Join-Path $SimDir "soc_repopt_two_layer.vvp"
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
    (Join-Path $TbDir "tb_soc_repopt_two_layer.v")
)

$args = @("-g2012", "-o", $VvpFile, "-s", "tb_soc_repopt_two_layer", "-I$CaseDir")
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
if (($simOut | Select-String "\[PASS\] RepOpt two-layer/stage1-pool SoC test PASSED").Count -eq 0) {
    exit 1
}

Write-Host "=== Done ===" -ForegroundColor Cyan
exit $simExit
