# =============================================================================
# run_conv2d_two_layer_case.ps1 - Generate and run one T6.6 Conv2D E2E case
#
# The generated testcase runs two direct scalar layers in one simulation. Layer0
# uses Conv2D on-the-fly im2col and layer1 consumes layer0 R_ADDR as A_ADDR.
# =============================================================================

param(
    [string]$Name = "conv2d_two_layer_int8_os_default"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir  = Join-Path $ProjectRoot "rtl"
$TbDir   = Join-Path $ProjectRoot "tb"
$SimDir  = Join-Path $ProjectRoot "sim"
$ConvDir = Join-Path $TbDir "conv2d"

if (-not (Test-Path $SimDir)) { New-Item -ItemType Directory -Path $SimDir | Out-Null }

Write-Host "Generating $Name (T6.6 two-layer Conv2D E2E)" -ForegroundColor Cyan
& python "$ConvDir\gen_conv2d_two_layer_data.py" --test-id $Name
if ($LASTEXITCODE -ne 0) { throw "Failed to generate T6.6 two-layer Conv2D data." }

$CaseDir = Join-Path $ConvDir $Name
$VvpOut = Join-Path $SimDir ("{0}.vvp" -f $Name)

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

Write-Host "Compiling $Name" -ForegroundColor Cyan
$compileArgs = @("-g2012", "-o", $VvpOut, "-I$CaseDir") + $RtlSrc + @("$ConvDir\tb_conv2d_two_layer.v")
$compileOut = & iverilog @compileArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host $compileOut -ForegroundColor Red
    throw "iverilog compile failed."
}

Write-Host "Running $Name" -ForegroundColor Cyan
Push-Location $CaseDir
$simOut = & vvp $VvpOut 2>&1
Pop-Location

$summary = ($simOut | Select-String "ALL .* CHECKS PASSED|RESULT: .*PASSED.*FAILED" | Select-Object -Last 1)
if ($summary) {
    Write-Host $summary.Line.Trim() -ForegroundColor Cyan
} else {
    Write-Host $simOut
}

$fail = ($simOut | Select-String "\[FAIL\]|FAILED|FATAL|TIMEOUT").Count
if ($fail -gt 0) {
    exit 1
}
