# =============================================================================
# run_repopt_tile_case.ps1 - Run one RepOpt VGG 4x4 tile-mode GEMM RTL case
#
# This validates ARR_CFG[7] tile mode with real RepOpt Conv2D data. The case is
# one local 4x4 GEMM tile: 4 im2col rows by 4 output channels.
# =============================================================================

param(
    [string]$Layer = "stage1_0_conv",
    [int]$Index = 0,
    [int]$MBase = 0,
    [int]$NBase = 0,
    [string]$Name = "",
    [string]$Pth = ".06_RepOpt_VGG\06_RepOpt_VGG\runs\cifar10_repopt_vgglike_qat\qat_int8_quantized.pth",
    [string]$Plan = "sim\pth_repopt_probe\model_plan.json",
    [string]$DataRoot = ".06_RepOpt_VGG\06_RepOpt_VGG\data",
    [switch]$DumpResult
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir = Join-Path $ProjectRoot "rtl"
$TbDir = Join-Path $ProjectRoot "tb"
$SimDir = Join-Path $ProjectRoot "sim"
$CaseRoot = Join-Path $SimDir "pth_repopt_tile_cases"

if (-not (Test-Path $SimDir)) { New-Item -ItemType Directory -Path $SimDir | Out-Null }
if (-not (Test-Path $CaseRoot)) { New-Item -ItemType Directory -Path $CaseRoot | Out-Null }

function Resolve-ProjectPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $ProjectRoot $PathValue)
}

if (-not $Name) {
    $Name = "repopt_${Layer}_tile4_m${MBase}_n${NBase}_idx${Index}"
}

$PthPath = Resolve-ProjectPath $Pth
$PlanPath = Resolve-ProjectPath $Plan
$DataRootPath = Resolve-ProjectPath $DataRoot

Write-Host "Generating $Name from RepOpt checkpoint for 4x4 tile mode" -ForegroundColor Cyan
& python "$ProjectRoot\tools\pth\gen_repopt_tile_case.py" `
    --pth "$PthPath" `
    --plan "$PlanPath" `
    --data-root "$DataRootPath" `
    --layer-name $Layer `
    --index $Index `
    --m-base $MBase `
    --n-base $NBase `
    --out-root "$CaseRoot" `
    --test-id $Name `
    --project-root "$ProjectRoot"
if ($LASTEXITCODE -ne 0) { throw "Failed to generate RepOpt tile case." }

$CaseDir = Join-Path $CaseRoot $Name
$TbFile = Join-Path $TbDir "tb_npu_tile_gemm.v"
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
$compileArgs = @("-g2012")
if ($DumpResult) { $compileArgs += "-DDUMP_RESULT_HEX" }
$compileArgs += @("-o", $VvpOut, "-I$CaseDir") + $RtlSrc + @($TbFile)
$compileOut = & iverilog @compileArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host $compileOut -ForegroundColor Red
    throw "iverilog compile failed."
}

Write-Host "Running $Name" -ForegroundColor Cyan
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
    $OutputHex = Join-Path $CaseDir "npu_output.hex"
    if (-not (Test-Path $OutputHex)) {
        throw "DumpResult requested but output file was not generated: $OutputHex"
    }
    Write-Host "Output: $OutputHex" -ForegroundColor Cyan
}
