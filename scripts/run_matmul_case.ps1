# =============================================================================
# run_matmul_case.ps1 - Generate and run one direct-mode matrix multiply case
#
# Examples:
#   powershell -ExecutionPolicy Bypass -File scripts\run_matmul_case.ps1 `
#     -M 32 -K 32 -N 32 -Dtype int8 -Mode OS
#
# This exercises the direct scalar matmul path. The 4x4 tile/descriptor path
# still expects pre-packed tile streams and has separate tests.
# =============================================================================

param(
    [int]$M = 16,
    [int]$K = 16,
    [int]$N = 16,
    [ValidateSet("int8", "fp16")]
    [string]$Dtype = "int8",
    [ValidateSet("OS", "WS")]
    [string]$Mode = "OS",
    [string]$Name = "",
    [switch]$Bias,
    [ValidateSet("none", "relu", "relu6")]
    [string]$Activation = "none",
    [switch]$Quant,
    [ValidateRange(-32768, 32767)]
    [int]$QuantScale = 1,
    [ValidateRange(0, 31)]
    [int]$QuantShift = 0,
    [switch]$QuantRound
)

$ErrorActionPreference = "Stop"

if ($M -le 0 -or $K -le 0 -or $N -le 0) {
    throw "M, K, and N must be positive."
}
if ($Quant -and $Dtype -ne "int8") {
    throw "T6.5 INT8 quant/saturate requires -Dtype int8."
}

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir  = Join-Path $ProjectRoot "rtl"
$TbDir   = Join-Path $ProjectRoot "tb"
$SimDir  = Join-Path $ProjectRoot "sim"
$MatmulDir = Join-Path $TbDir "matmul"

if (-not (Test-Path $SimDir)) { New-Item -ItemType Directory -Path $SimDir | Out-Null }

if (-not $Name) {
    $Name = ("custom_{0}_{1}_{2}x{3}x{4}" -f $Dtype, $Mode.ToLower(), $M, $K, $N)
    if ($Bias) { $Name = "${Name}_bias" }
    if ($Activation -ne "none") { $Name = "${Name}_${Activation}" }
    if ($Quant) {
        $Name = "${Name}_quant_s${QuantScale}_sh${QuantShift}"
        if ($QuantRound) { $Name = "${Name}_rnd" }
    }
}

Write-Host "Generating $Name ($Dtype $Mode, M=$M K=$K N=$N)" -ForegroundColor Cyan
$BiasArg = @()
if ($Bias) { $BiasArg = @("--bias") }
$QuantArg = @()
if ($Quant) { $QuantArg = @("--quant") }
if ($QuantRound) { $QuantArg += "--quant-round" }
$GenArgs = @("--custom", "--m", $M, "--k", $K, "--n", $N,
             "--dtype", $Dtype, "--mode", $Mode, "--test-id", $Name,
             "--activation", $Activation,
             "--quant-scale", $QuantScale, "--quant-shift", $QuantShift) + $BiasArg + $QuantArg
& python "$MatmulDir\gen_matmul_data.py" @GenArgs
if ($LASTEXITCODE -ne 0) { throw "Failed to generate matmul data." }

$CaseDir = Join-Path $MatmulDir $Name
$TbFile = if ($Mode -eq "WS") { "$MatmulDir\tb_matmul_ws.v" } else { "$MatmulDir\tb_matmul_os.v" }
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
$compileArgs = @("-g2012", "-o", $VvpOut, "-I$CaseDir") + $RtlSrc + @($TbFile)
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

$fail = ($simOut | Select-String "\[FAIL\]|FAILED|FATAL").Count
if ($fail -gt 0) {
    exit 1
}
