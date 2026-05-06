# =============================================================================
# run_image_conv_case.ps1 - Visual image Conv2D demo for NPU simulation
#
# Generates a signed-INT8 grayscale image Conv2D case, runs the existing
# direct scalar on-the-fly im2col simulation, dumps the NPU result, and renders
# input/golden/NPU/diff PNGs.
# =============================================================================

param(
    [string]$ImagePath = "pic\test2_128.png",
    [ValidateSet("laplacian", "sobel_x", "sobel_y", "sharpen")]
    [string]$Kernel = "laplacian",
    [ValidateSet("OS", "WS")]
    [string]$Mode = "OS",
    [string]$Name = "",
    [int]$Resize = 0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir  = Join-Path $ProjectRoot "rtl"
$TbDir   = Join-Path $ProjectRoot "tb"
$SimDir  = Join-Path $ProjectRoot "sim"
$ImageTbDir = Join-Path $TbDir "image"
$MatmulDir = Join-Path $TbDir "matmul"

if (-not (Test-Path $SimDir)) { New-Item -ItemType Directory -Path $SimDir | Out-Null }

if ([System.IO.Path]::IsPathRooted($ImagePath)) {
    $ResolvedImage = Resolve-Path $ImagePath
} else {
    $ResolvedImage = Resolve-Path (Join-Path $ProjectRoot $ImagePath)
}
if (-not $Name) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($ResolvedImage.Path)
    $Name = "image_${Kernel}_${stem}"
}

Write-Host "Generating image Conv2D case: $Name" -ForegroundColor Cyan
$genArgs = @(
    "$ImageTbDir\gen_image_conv_data.py",
    "--image", $ResolvedImage.Path,
    "--kernel", $Kernel,
    "--mode", $Mode,
    "--test-id", $Name,
    "--output-root", $ImageTbDir
)
if ($Resize -gt 0) {
    $genArgs += @("--resize", $Resize)
}
& python @genArgs
if ($LASTEXITCODE -ne 0) { throw "Failed to generate image Conv2D data." }

$CaseDir = Join-Path $ImageTbDir $Name
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

Write-Host "Compiling image Conv2D simulation" -ForegroundColor Cyan
$compileArgs = @(
    "-g2012",
    "-DQUIET_CHECK",
    "-DDUMP_RESULT_HEX",
    "-o", $VvpOut,
    "-I$ProjectRoot",
    "-I$CaseDir"
) + $RtlSrc + @($TbFile)
$compileOut = & iverilog @compileArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host $compileOut -ForegroundColor Red
    throw "iverilog compile failed."
}

Write-Host "Running image Conv2D simulation" -ForegroundColor Cyan
Push-Location $CaseDir
$simOut = & vvp $VvpOut 2>&1
$vvpExit = $LASTEXITCODE
Pop-Location

$summary = ($simOut | Select-String "ALL .* CHECKS PASSED|RESULT: .*PASSED.*FAILED|\\[DUMP\\]" | Select-Object -Last 4)
if ($summary) {
    $summary | ForEach-Object { Write-Host $_.Line.Trim() -ForegroundColor Cyan }
} else {
    Write-Host $simOut
}

$fail = ($simOut | Select-String "\[FAIL\]|FAILED|FATAL|TIMEOUT").Count
if ($vvpExit -ne 0 -or $fail -gt 0) {
    Write-Host $simOut -ForegroundColor Red
    exit 1
}

Write-Host "Rendering PNG outputs" -ForegroundColor Cyan
& python "$ImageTbDir\gen_image_conv_data.py" --render-only --case-dir $CaseDir
if ($LASTEXITCODE -ne 0) { throw "Failed to render image Conv2D outputs." }

Write-Host ""
Write-Host "Image Conv2D visual outputs:" -ForegroundColor Green
Write-Host "  Case dir : $CaseDir"
Write-Host "  Input    : $(Join-Path $CaseDir 'input_gray.png')"
Write-Host "  Golden   : $(Join-Path $CaseDir 'golden.png')"
Write-Host "  NPU      : $(Join-Path $CaseDir 'npu.png')"
Write-Host "  Diff     : $(Join-Path $CaseDir 'diff.png')"
Write-Host "  Compare  : $(Join-Path $CaseDir 'comparison.png')"
