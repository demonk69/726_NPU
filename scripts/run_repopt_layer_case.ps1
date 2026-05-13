# =============================================================================
# run_repopt_layer_case.ps1 - Run one staged RepOpt VGG Conv2D RTL case
#
# Generates a real-checkpoint Conv/ReLU case and runs it through npu_top using
# the existing tb/matmul/tb_matmul_os.v direct Conv2D path.
# =============================================================================

param(
    [string]$Layer = "stage1_0_conv",
    [int]$Index = 0,
    [int]$TileOH = 4,
    [int]$TileOW = 4,
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
$MatmulDir = Join-Path $TbDir "matmul"
$CaseRoot = Join-Path $SimDir "pth_repopt_layer_cases"

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
    $TileSuffix = ""
    if ($TileOH -gt 0 -and $TileOW -gt 0) {
        $TileSuffix = "_tile${TileOH}x${TileOW}"
    }
    $Name = "repopt_${Layer}_idx${Index}${TileSuffix}"
}

$PthPath = Resolve-ProjectPath $Pth
$PlanPath = Resolve-ProjectPath $Plan
$DataRootPath = Resolve-ProjectPath $DataRoot

Write-Host "Generating $Name from RepOpt checkpoint" -ForegroundColor Cyan
& python "$ProjectRoot\tools\pth\gen_repopt_layer_case.py" `
    --pth "$PthPath" `
    --plan "$PlanPath" `
    --data-root "$DataRootPath" `
    --layer-name $Layer `
    --index $Index `
    --tile-oh $TileOH `
    --tile-ow $TileOW `
    --out-root "$CaseRoot" `
    --test-id $Name
if ($LASTEXITCODE -ne 0) { throw "Failed to generate RepOpt layer case." }

$CaseDir = Join-Path $CaseRoot $Name
$TbFile = Join-Path $MatmulDir "tb_matmul_os.v"
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
$compileArgs = @("-g2012", "-DQUIET_CHECK")
if ($DumpResult) { $compileArgs += "-DDUMP_RESULT_HEX" }
$compileArgs += @("-o", $VvpOut, "-I$CaseDir") + $RtlSrc + @($TbFile)
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
