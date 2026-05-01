# =============================================================================
# run_conv2d_im2col_case.ps1 - Generate and run one T6.1 Conv2D im2col case
#
# The Conv2D layer is expanded into DRAM-resident A_im2col and W_col matrices,
# then checked with the existing direct-mode matmul testbench against Conv2D
# golden output.
# =============================================================================

param(
    [int]$Batch = 1,
    [int]$IH = 5,
    [int]$IW = 5,
    [int]$Cin = 2,
    [int]$Cout = 3,
    [int]$KH = 3,
    [int]$KW = 3,
    [int]$StrideH = 1,
    [int]$StrideW = 1,
    [int]$PadH = 1,
    [int]$PadW = 1,
    [int]$DilationH = 1,
    [int]$DilationW = 1,
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

if ($Batch -le 0 -or $IH -le 0 -or $IW -le 0 -or $Cin -le 0 -or $Cout -le 0 -or
    $KH -le 0 -or $KW -le 0 -or $StrideH -le 0 -or $StrideW -le 0 -or
    $DilationH -le 0 -or $DilationW -le 0) {
    throw "Batch, dimensions, strides, and dilations must be positive."
}
if ($PadH -lt 0 -or $PadW -lt 0) {
    throw "Padding must be non-negative."
}
if ($Quant -and $Dtype -ne "int8") {
    throw "T6.5 INT8 quant/saturate requires -Dtype int8."
}

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir  = Join-Path $ProjectRoot "rtl"
$TbDir   = Join-Path $ProjectRoot "tb"
$SimDir  = Join-Path $ProjectRoot "sim"
$ConvDir = Join-Path $TbDir "conv2d"
$MatmulDir = Join-Path $TbDir "matmul"

if (-not (Test-Path $SimDir)) { New-Item -ItemType Directory -Path $SimDir | Out-Null }

if (-not $Name) {
    $Name = ("conv2d_im2col_{0}_{1}_b{2}_{3}x{4}_c{5}_k{6}x{7}_co{8}_p{9}x{10}" -f `
        $Dtype, $Mode.ToLower(), $Batch, $IH, $IW, $Cin, $KH, $KW, $Cout, $PadH, $PadW)
    if ($Bias) { $Name = "${Name}_bias" }
    if ($Activation -ne "none") { $Name = "${Name}_${Activation}" }
    if ($Quant) {
        $Name = "${Name}_quant_s${QuantScale}_sh${QuantShift}"
        if ($QuantRound) { $Name = "${Name}_rnd" }
    }
}

Write-Host "Generating $Name ($Dtype $Mode Conv2D im2col)" -ForegroundColor Cyan
$BiasArg = @()
if ($Bias) { $BiasArg = @("--bias") }
$QuantArg = @()
if ($Quant) { $QuantArg = @("--quant") }
if ($QuantRound) { $QuantArg += "--quant-round" }
& python "$ConvDir\gen_conv2d_im2col_data.py" `
    --batch $Batch --ih $IH --iw $IW --cin $Cin --cout $Cout `
    --kh $KH --kw $KW --stride-h $StrideH --stride-w $StrideW `
    --pad-h $PadH --pad-w $PadW --dilation-h $DilationH --dilation-w $DilationW `
    --dtype $Dtype --mode $Mode --test-id $Name --activation $Activation `
    --quant-scale $QuantScale --quant-shift $QuantShift @BiasArg @QuantArg
if ($LASTEXITCODE -ne 0) { throw "Failed to generate Conv2D im2col data." }

$CaseDir = Join-Path $ConvDir $Name
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

$fail = ($simOut | Select-String "\[FAIL\]|FAILED|FATAL|TIMEOUT").Count
if ($fail -gt 0) {
    exit 1
}
