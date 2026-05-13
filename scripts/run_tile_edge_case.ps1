# =============================================================================
# run_tile_edge_case.ps1 - Generate and run one P2 tile edge smoke case
#
# Current RTL smoke path is still 4x4 tile GEMM compatible. The script accepts
# shape/mode/lane parameters, generates the matching tile-edge data set, and
# runs the existing 4x4 tile GEMM testbench when the combination is supported.
# Other combinations are generated and then reported as not yet connected to
# the full RTL smoke path.
# =============================================================================

param(
    [ValidateSet("4x4", "8x8", "16x16", "8x32")]
    [string]$Shape = "4x4",
    [ValidateSet("OS", "WS")]
    [string]$Mode = "OS",
    [ValidateSet(1, 2, 4)]
    [int]$Lanes = 1,
    [string]$CaseName = "",
    [switch]$DumpResult
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RtlDir = Join-Path $ProjectRoot "rtl"
$TbDir = Join-Path $ProjectRoot "tb"
$SimDir = Join-Path $ProjectRoot "sim"
$EdgeGen = Join-Path $ProjectRoot "scripts\gen_tile_edge_cases.ps1"

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
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not found. Add Python to PATH first."
}

if (-not (Test-Path $SimDir)) {
    New-Item -ItemType Directory -Path $SimDir | Out-Null
}

if (-not $CaseName) {
    $CaseName = "edge_${Shape}_$($Mode.ToLower())_l${Lanes}_smoke"
}

Write-Host "=== P2 tile edge smoke ===" -ForegroundColor Cyan
Write-Host "[1/3] Generating $CaseName ..." -ForegroundColor Yellow
& powershell -NoProfile -ExecutionPolicy Bypass -File $EdgeGen `
    -Shape $Shape `
    -Mode $Mode `
    -Lanes $Lanes `
    -Name $CaseName
if ($LASTEXITCODE -ne 0) {
    throw "tile edge generation failed."
}

$CaseDir = Join-Path $TbDir "edge\$CaseName"
if (-not (Test-Path $CaseDir)) {
    throw "generated case directory not found: $CaseDir"
}

$SupportedRun = (($Shape -eq "4x4") -or ($Shape -eq "8x8") -or ($Shape -eq "16x16")) -and ($Lanes -eq 1)
if (-not $SupportedRun) {
    Write-Host "[2/3] RTL smoke path is not yet wired for $Shape / $Mode / lanes=$Lanes." -ForegroundColor Yellow
    Write-Host "      Generated data is ready at $CaseDir" -ForegroundColor Cyan
    Write-Host "      Current RTL smoke covers 4x4, 8x8, and 16x16 single-lane OS tile edge cases only." -ForegroundColor Yellow
    exit 0
}

$VvpOut = Join-Path $SimDir ("{0}.vvp" -f $CaseName)
$RtlSrc = @(
    "$RtlDir\pe\fp16_mul.v",
    "$RtlDir\pe\fp16_add.v",
    "$RtlDir\pe\fp32_add.v",
    "$RtlDir\pe\pe_top.v",
    "$RtlDir\common\fifo.v",
    "$RtlDir\common\axi_monitor.v",
    "$RtlDir\common\op_counter.v",
    "$RtlDir\buf\pingpong_buf.v",
    "$RtlDir\buf\psum_out_buf.v",
    "$RtlDir\array\reconfig_pe_array.v",
    "$RtlDir\power\npu_power.v",
    "$RtlDir\ctrl\npu_ctrl.v",
    "$RtlDir\axi\npu_axi_lite.v",
    "$RtlDir\axi\npu_dma.v",
    "$RtlDir\top\npu_top.v"
)

Write-Host "[2/3] Compiling..." -ForegroundColor Yellow
$compileArgs = @("-g2012", "-o", $VvpOut, "-I$CaseDir")
if ($DumpResult) {
    $compileArgs += "-DDUMP_RESULT_HEX"
    $OutputHex = Join-Path $CaseDir "npu_output.hex"
    $OutputHexDefine = $OutputHex.Replace("\", "/")
    $OutputDefineVh = Join-Path $CaseDir "output_hex_define.vh"
    Set-Content -Path $OutputDefineVh -Encoding ASCII -Value ("``define OUTPUT_HEX `"$OutputHexDefine`"")
    $compileArgs += "-DOUTPUT_HEX_VH"
}
$compileArgs += $RtlSrc + @("$TbDir\tb_npu_tile_gemm.v")
$compileOut = & iverilog @compileArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host $compileOut -ForegroundColor Red
    throw "iverilog compile failed."
}

Write-Host "[3/3] Running simulation..." -ForegroundColor Yellow
Push-Location $ProjectRoot
$simOut = & vvp $VvpOut 2>&1
Pop-Location

$summary = ($simOut | Select-String "\[PASS\]|\[FAIL\]|ALL .* CHECKS PASSED|RESULT: .*PASSED.*FAILED" | Select-Object -Last 1)
if ($summary) {
    Write-Host $summary.Line.Trim() -ForegroundColor Cyan
} else {
    Write-Host $simOut
}

$fail = ($simOut | Select-String "\[FAIL\]|FAILED|FATAL|TIMEOUT").Count
if ($fail -gt 0) {
    exit 1
}

if ($DumpResult) {
    if (-not (Test-Path $OutputHex)) {
        throw "DumpResult requested but output file was not generated: $OutputHex"
    }
    Write-Host "Output: $OutputHex" -ForegroundColor Cyan
}
