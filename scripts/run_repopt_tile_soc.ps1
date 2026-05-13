# =============================================================================
# run_repopt_tile_soc.ps1 - Run RepOpt tile-window SoC MMIO scheduling smoke.
#
# The generated firmware runs on the reference CPU, schedules multiple
# ARR_CFG[7] tile-mode NPU jobs in one RTL simulation, and postprocesses them.
# =============================================================================

param(
    [int]$Index = 0,
    [int]$MBase = 0,
    [int]$NBase = 0,
    [int]$MTiles = 2,
    [int]$NTiles = 2,
    [switch]$FullLayer,
    [switch]$CompileOnly,
    [switch]$DumpVcd
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RtlDir = Join-Path $ProjectRoot "rtl"
$TbDir = Join-Path $ProjectRoot "tb"
$SimDir = Join-Path $ProjectRoot "sim"
$Picorv32Dir = Join-Path $ProjectRoot "picorv32_ref"
$CaseDir = Join-Path $SimDir "repopt_tile_soc"

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

Write-Host "=== RepOpt Tile-Window SoC MMIO Simulation ===" -ForegroundColor Cyan
Write-Host "[1/3] Generating RepOpt tile DRAM image and CPU firmware..." -ForegroundColor Yellow
$genArgs = @(
    (Join-Path $ProjectRoot "tools\pth\gen_repopt_tile_soc_case.py"),
    "--out-dir", $CaseDir,
    "--index", $Index,
    "--m-base", $MBase,
    "--n-base", $NBase,
    "--m-tiles", $MTiles,
    "--n-tiles", $NTiles
)
if ($FullLayer) {
    $genArgs += "--full-layer"
}
& $PythonCmd.Source @genArgs
if ($LASTEXITCODE -ne 0) {
    throw "RepOpt tile SoC case generation failed."
}

Copy-Item (Join-Path $Picorv32Dir "picorv32.v") (Join-Path $SimDir "picorv32.v") -Force

Write-Host "[2/3] Compiling Verilog..." -ForegroundColor Yellow
$VvpFile = Join-Path $SimDir "soc_repopt_tile_window.vvp"
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
    (Join-Path $TbDir "tb_soc_repopt_tile_window.v")
)

$args = @("-g2012", "-o", $VvpFile, "-s", "tb_soc_repopt_tile_window", "-I$CaseDir")
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
if ($CompileOnly) {
    Write-Host "=== CompileOnly requested; simulation skipped ===" -ForegroundColor Cyan
    exit 0
}

Write-Host "[3/3] Running simulation..." -ForegroundColor Yellow
Push-Location $SimDir
$simOut = & vvp -N $VvpFile 2>&1
$simExit = $LASTEXITCODE
Pop-Location

$simOut | ForEach-Object { Write-Host $_ }

if (($simOut | Select-String "\[FAIL\]|\[TIMEOUT\]|FATAL|ERROR:").Count -gt 0) {
    exit 1
}
if (($simOut | Select-String "\[PASS\] RepOpt tile-window SoC MMIO \+ CPU postprocess test PASSED").Count -eq 0) {
    exit 1
}

Write-Host "=== Done ===" -ForegroundColor Cyan
exit $simExit
