# =============================================================================
# run_regression.ps1 - Full regression suite for NPU_prj
#
# Rebuilds and runs all OS and WS matmul test cases from source RTL.
# Also runs: tb_fp16_e2e, tb_multi_rc_comprehensive
#
# Prerequisites: iverilog, vvp
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RtlDir  = Join-Path $ProjectRoot "rtl"
$TbDir   = Join-Path $ProjectRoot "tb"
$SimDir  = Join-Path $ProjectRoot "sim"

if (-not (Test-Path $SimDir)) { New-Item -ItemType Directory -Path $SimDir | Out-Null }

# Common RTL sources
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

$TotalPass = 0
$TotalFail = 0
$Results   = @()

function Run-Test {
    param(
        [string]$Name,
        [string]$VvpOut,
        [string[]]$ExtraSrc,
        [string]$IncDir = "",
        [string]$RunDir = "",
        [int]$ExpectedPass = -1
    )
    Write-Host ""
    Write-Host "--- [$Name] ---" -ForegroundColor Cyan

    $args = @("-g2012", "-o", $VvpOut) + $RtlSrc + $ExtraSrc
    if ($IncDir) { $args = @("-g2012", "-o", $VvpOut, "-I$IncDir") + $RtlSrc + $ExtraSrc }

    $compOut = & iverilog @args 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [COMPILE FAIL] $compOut" -ForegroundColor Red
        $script:TotalFail++
        $script:Results += [PSCustomObject]@{ Name=$Name; Status="COMPILE FAIL"; Pass=0; Fail=1 }
        return
    }

    if (-not $RunDir) { $RunDir = $ProjectRoot }
    Push-Location $RunDir
    $simOut = & vvp $VvpOut 2>&1
    Pop-Location

    $pass = ($simOut | Select-String "\[PASS\]").Count
    $fail = ($simOut | Select-String "\[FAIL\]").Count
    $timeout = ($simOut | Select-String "TIMEOUT").Count
    $simError = ($simOut | Select-String "ERROR:").Count

    # Also check summary lines (matmul testbench style: "ALL N CHECKS PASSED")
    $summaryLine = ($simOut | Select-String "RESULT:|ALL.*PASS|ALL.*CHECKS PASSED|PASSED.*FAILED" | Select-Object -Last 1)
    if ($summaryLine) { Write-Host "  $($summaryLine.Line.Trim())" -ForegroundColor Cyan }

    # Prefer explicit testbench summary counts when present.
    $chkMatch = ($simOut | Select-String "ALL (\d+) CHECKS PASSED" | Select-Object -Last 1)
    if ($chkMatch) {
        $pass = [int]$chkMatch.Matches[0].Groups[1].Value
        $fail = 0
    }
    $r2Match = ($simOut | Select-String "(\d+) PASSED.*?(\d+) FAILED" | Select-Object -Last 1)
    if ($r2Match) {
        $pass = [int]$r2Match.Matches[0].Groups[1].Value
        $fail = [int]$r2Match.Matches[0].Groups[2].Value
    }

    if ($timeout -gt 0) {
        $fail += $timeout
        Write-Host "  [TIMEOUT detected]" -ForegroundColor Red
    }
    if ($simError -gt 0) {
        $fail += $simError
        Write-Host "  [Simulator ERROR detected]" -ForegroundColor Red
    }

    $status = if ($fail -eq 0 -and $pass -gt 0) { "PASS" } elseif ($fail -gt 0) { "FAIL" } else { "UNKNOWN" }
    $color  = if ($status -eq "PASS") { "Green" } else { "Red" }

    Write-Host "  Result: $pass PASS, $fail FAIL  -> $status" -ForegroundColor $color

    $script:TotalPass += $pass
    $script:TotalFail += $fail
    $script:Results += [PSCustomObject]@{ Name=$Name; Status=$status; Pass=$pass; Fail=$fail }
}

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  NPU Full Regression Suite" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# ---- 1. FP16 E2E ----
Run-Test -Name "fp16_e2e" `
    -VvpOut "$SimDir\reg_fp16_e2e.vvp" `
    -ExtraSrc @("$TbDir\tb_fp16_e2e.v")

# ---- 2. Multi-RC Comprehensive ----
Run-Test -Name "multi_rc_comprehensive" `
    -VvpOut "$SimDir\reg_multi_rc.vvp" `
    -ExtraSrc @("$TbDir\tb_multi_rc_comprehensive.v")

# ---- 2a0. DMA Read Burst ----
Run-Test -Name "dma_read_burst" `
    -VvpOut "$SimDir\reg_dma_read_burst.vvp" `
    -ExtraSrc @("$TbDir\tb_dma_read_burst.v")

# ---- 2a1. DMA Write Burst ----
Run-Test -Name "dma_write_burst" `
    -VvpOut "$SimDir\reg_dma_write_burst.vvp" `
    -ExtraSrc @("$TbDir\tb_dma_write_burst.v")

# ---- 2a2. DMA Mixed Burst Correctness ----
Run-Test -Name "dma_burst" `
    -VvpOut "$SimDir\reg_dma_burst.vvp" `
    -ExtraSrc @("$TbDir\tb_dma_burst.v")

# ---- 2a3. DMA Bandwidth Utilization ----
Run-Test -Name "dma_perf" `
    -VvpOut "$SimDir\reg_dma_perf.vvp" `
    -ExtraSrc @("$TbDir\tb_dma_perf.v")

# ---- 2a4. PSUM/OUT Buffer RMW ----
Run-Test -Name "psum_out_buf" `
    -VvpOut "$SimDir\reg_psum_out_buf.vvp" `
    -ExtraSrc @("$TbDir\tb_psum_out_buf.v")

# ---- 2a5. PE Array Accumulator Init ----
Run-Test -Name "reconfig_pe_acc_init" `
    -VvpOut "$SimDir\reg_reconfig_pe_acc_init.vvp" `
    -ExtraSrc @("$TbDir\tb_reconfig_pe_acc_init.v")

# ---- 2a6. Controller K-split Loop ----
Run-Test -Name "npu_ctrl_ksplit" `
    -VvpOut "$SimDir\reg_npu_ctrl_ksplit.vvp" `
    -ExtraSrc @("$TbDir\tb_npu_ctrl_ksplit.v")

# ---- 2a6a. Controller OS/WS direct dataflow branches ----
Run-Test -Name "npu_ctrl_dataflow_modes" `
    -VvpOut "$SimDir\reg_npu_ctrl_dataflow_modes.vvp" `
    -ExtraSrc @("$TbDir\tb_npu_ctrl_dataflow_modes.v")

# ---- 2a6b. Controller descriptor error status ----
Run-Test -Name "npu_ctrl_error_status" `
    -VvpOut "$SimDir\reg_npu_ctrl_error_status.vvp" `
    -ExtraSrc @("$TbDir\tb_npu_ctrl_error_status.v")

# ---- 2a7. End-to-end Tile K-split GEMM ----
Run-Test -Name "npu_tile_ksplit_gemm" `
    -VvpOut "$SimDir\reg_npu_tile_ksplit_gemm.vvp" `
    -ExtraSrc @("$TbDir\tb_npu_tile_ksplit_gemm.v")

# ---- 2a8. AXI-Lite descriptor submission registers ----
Run-Test -Name "npu_axi_lite_desc" `
    -VvpOut "$SimDir\reg_npu_axi_lite_desc.vvp" `
    -ExtraSrc @("$TbDir\tb_npu_axi_lite_desc.v")

# ---- 2a9. Descriptor fetch/decode two-layer sequence ----
Run-Test -Name "npu_desc_two_layer" `
    -VvpOut "$SimDir\reg_npu_desc_two_layer.vvp" `
    -ExtraSrc @("$TbDir\tb_npu_desc_two_layer.v")

# ---- 2a10. Descriptor OFM->IFM chained GEMM ----
Run-Test -Name "npu_desc_ofm_chain" `
    -VvpOut "$SimDir\reg_npu_desc_ofm_chain.vvp" `
    -ExtraSrc @("$TbDir\tb_npu_desc_ofm_chain.v")

# ---- 2a10. Top scalar smoke + perf counters ----
Run-Test -Name "npu_scalar_smoke" `
    -VvpOut "$SimDir\reg_npu_scalar_smoke.vvp" `
    -ExtraSrc @("$TbDir\tb_npu_scalar_smoke.v")

# ---- 2a. 4x4 Tile Writeback ----
Run-Test -Name "npu_tile_writeback" `
    -VvpOut "$SimDir\reg_npu_tile_writeback.vvp" `
    -ExtraSrc @("$TbDir\tb_npu_tile_writeback.v")

# ---- 2b. 4x4 Tile GEMM ----
# Each case sets M=N=K=4 in test_params.vh.
# The input hex files contain pre-packed A_TILE[k][r] and W_TILE[k][c] streams.
Run-Test -Name "tile4/int8_4x4x4" `
    -VvpOut "$SimDir\reg_tile4_int8_4x4x4.vvp" `
    -IncDir "$TbDir\tile4\int8_4x4x4" `
    -ExtraSrc @("$TbDir\tb_npu_tile_gemm.v")

Run-Test -Name "tile4/fp16_4x4x4" `
    -VvpOut "$SimDir\reg_tile4_fp16_4x4x4.vvp" `
    -IncDir "$TbDir\tile4\fp16_4x4x4" `
    -ExtraSrc @("$TbDir\tb_npu_tile_gemm.v")

# ---- 3. OS Matmul tests ----
$OsCases = @("os_int8_2x3x2", "os_int8_2x4x3", "os_int8_3x4x3", "os_fp16_2x3x2", "os_fp16_3x4x3",
             "sq_int8_4x4", "sq_int8_8x8", "sq_int8_16x16", "sq_fp16_4x4", "sq_fp16_8x8")

foreach ($tc in $OsCases) {
    $tcDir = "$TbDir\matmul\$tc"
    if (-not (Test-Path "$tcDir\dram_init.hex")) {
        Write-Host "  [SKIP] $tc (data not found)" -ForegroundColor Yellow
        continue
    }
    Run-Test -Name "os/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @("$TbDir\matmul\tb_matmul_os.v")
}

# ---- 4. WS Matmul tests ----
$WsCases = @("ws_int8_2x3x2", "ws_int8_3x4x3", "ws_fp16_2x3x2",
             "ws_sq_int8_4x4", "ws_sq_int8_8x8", "ws_sq_int8_16x16", "ws_sq_fp16_4x4", "ws_sq_fp16_8x8")

# Check if tb_matmul_ws.v exists, else skip
$WsTb = "$TbDir\matmul\tb_matmul_ws.v"
if (-not (Test-Path $WsTb)) {
    $WsTb = "$TbDir\matmul\tb_matmul_os.v"
    Write-Host "`n[INFO] tb_matmul_ws.v not found, using tb_matmul_os.v for WS cases" -ForegroundColor Yellow
}

foreach ($tc in $WsCases) {
    $tcDir = "$TbDir\matmul\$tc"
    if (-not (Test-Path "$tcDir\dram_init.hex")) {
        Write-Host "  [SKIP] $tc (data not found)" -ForegroundColor Yellow
        continue
    }
    Run-Test -Name "ws/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @($WsTb)
}

# ---- 5. T6.1 Conv2D im2col tests ----
# These cases keep im2col pre-expanded in DRAM/test data and reuse the direct
# matmul checker with Conv2D golden expected output.
$ConvDir = "$TbDir\conv2d"
$ConvCases = @(
    @{ Name="conv2d_im2col_int8_os_default"; Dtype="int8"; Mode="OS"; Tb="$TbDir\matmul\tb_matmul_os.v" },
    @{ Name="conv2d_im2col_int8_ws_default"; Dtype="int8"; Mode="WS"; Tb=$WsTb },
    @{ Name="conv2d_im2col_fp16_os_default"; Dtype="fp16"; Mode="OS"; Tb="$TbDir\matmul\tb_matmul_os.v" }
)

foreach ($case in $ConvCases) {
    $tc = $case.Name
    Write-Host ""
    Write-Host "--- [generate conv2d_im2col/$tc] ---" -ForegroundColor Cyan
    $genOut = & python "$ConvDir\gen_conv2d_im2col_data.py" `
        --dtype $case.Dtype --mode $case.Mode --test-id $tc 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [GEN FAIL] $genOut" -ForegroundColor Red
        $TotalFail++
        $Results += [PSCustomObject]@{ Name="conv2d_im2col/$tc"; Status="GEN FAIL"; Pass=0; Fail=1 }
        continue
    }

    $tcDir = "$ConvDir\$tc"
    Run-Test -Name "conv2d_im2col/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @($case.Tb)
}

# ---- 6. T6.2 Conv2D on-the-fly im2col tests ----
# These cases keep only raw NCHW IFM in DRAM. DMA gathers A_im2col rows on the
# fly from the Conv2D shape registers while W_col remains packed in DRAM.
$ConvOtfCases = @(
    @{ Name="conv2d_otf_int8_os_default"; Dtype="int8"; Mode="OS"; Tb="$TbDir\matmul\tb_matmul_os.v" },
    @{ Name="conv2d_otf_int8_ws_default"; Dtype="int8"; Mode="WS"; Tb=$WsTb },
    @{ Name="conv2d_otf_fp16_os_default"; Dtype="fp16"; Mode="OS"; Tb="$TbDir\matmul\tb_matmul_os.v" }
)

foreach ($case in $ConvOtfCases) {
    $tc = $case.Name
    Write-Host ""
    Write-Host "--- [generate conv2d_otf/$tc] ---" -ForegroundColor Cyan
    $genOut = & python "$ConvDir\gen_conv2d_im2col_data.py" `
        --on-the-fly --dtype $case.Dtype --mode $case.Mode --test-id $tc 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [GEN FAIL] $genOut" -ForegroundColor Red
        $TotalFail++
        $Results += [PSCustomObject]@{ Name="conv2d_otf/$tc"; Status="GEN FAIL"; Pass=0; Fail=1 }
        continue
    }

    $tcDir = "$ConvDir\$tc"
    Run-Test -Name "conv2d_otf/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @($case.Tb)
}

# ---- 7. T6.3 Bias addition tests ----
# Bias is a 32-bit per-output-channel vector read through BIAS_ADDR and added
# to the direct scalar accumulator before each output dot product.
$MatmulBiasCases = @(
    @{ Name="matmul_bias_int8_os_3x5x4"; Dtype="int8"; Mode="OS"; M=3; K=5; N=4; Tb="$TbDir\matmul\tb_matmul_os.v" },
    @{ Name="matmul_bias_int8_ws_3x5x4"; Dtype="int8"; Mode="WS"; M=3; K=5; N=4; Tb=$WsTb },
    @{ Name="matmul_bias_fp16_os_3x4x3"; Dtype="fp16"; Mode="OS"; M=3; K=4; N=3; Tb="$TbDir\matmul\tb_matmul_os.v" },
    @{ Name="matmul_bias_fp16_ws_2x3x2"; Dtype="fp16"; Mode="WS"; M=2; K=3; N=2; Tb=$WsTb }
)

foreach ($case in $MatmulBiasCases) {
    $tc = $case.Name
    Write-Host ""
    Write-Host "--- [generate matmul_bias/$tc] ---" -ForegroundColor Cyan
    $genOut = & python "$TbDir\matmul\gen_matmul_data.py" --custom `
        --m $case.M --k $case.K --n $case.N `
        --dtype $case.Dtype --mode $case.Mode --bias --test-id $tc 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [GEN FAIL] $genOut" -ForegroundColor Red
        $TotalFail++
        $Results += [PSCustomObject]@{ Name="matmul_bias/$tc"; Status="GEN FAIL"; Pass=0; Fail=1 }
        continue
    }

    $tcDir = "$TbDir\matmul\$tc"
    Run-Test -Name "matmul_bias/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @($case.Tb)
}

$ConvBiasCases = @(
    @{ Name="conv2d_bias_otf_int8_os_default"; Dtype="int8"; Mode="OS"; Tb="$TbDir\matmul\tb_matmul_os.v" },
    @{ Name="conv2d_bias_otf_int8_ws_default"; Dtype="int8"; Mode="WS"; Tb=$WsTb },
    @{ Name="conv2d_bias_otf_fp16_os_default"; Dtype="fp16"; Mode="OS"; Tb="$TbDir\matmul\tb_matmul_os.v" }
)

foreach ($case in $ConvBiasCases) {
    $tc = $case.Name
    Write-Host ""
    Write-Host "--- [generate conv2d_bias_otf/$tc] ---" -ForegroundColor Cyan
    $genOut = & python "$ConvDir\gen_conv2d_im2col_data.py" `
        --on-the-fly --bias --dtype $case.Dtype --mode $case.Mode --test-id $tc 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [GEN FAIL] $genOut" -ForegroundColor Red
        $TotalFail++
        $Results += [PSCustomObject]@{ Name="conv2d_bias_otf/$tc"; Status="GEN FAIL"; Pass=0; Fail=1 }
        continue
    }

    $tcDir = "$ConvDir\$tc"
    Run-Test -Name "conv2d_bias_otf/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @($case.Tb)
}

# ---- 8. T6.4 ReLU/ReLU6 activation tests ----
# Direct scalar postprocess order is accumulator -> optional bias -> activation.
$MatmulActCases = @(
    @{ Name="matmul_relu_int8_os_3x5x4"; Dtype="int8"; Mode="OS"; M=3; K=5; N=4; Act="relu"; Tb="$TbDir\matmul\tb_matmul_os.v" },
    @{ Name="matmul_relu6_int8_ws_3x5x4"; Dtype="int8"; Mode="WS"; M=3; K=5; N=4; Act="relu6"; Tb=$WsTb },
    @{ Name="matmul_relu_fp16_os_3x4x3"; Dtype="fp16"; Mode="OS"; M=3; K=4; N=3; Act="relu"; Tb="$TbDir\matmul\tb_matmul_os.v" },
    @{ Name="matmul_relu6_fp16_ws_2x3x2"; Dtype="fp16"; Mode="WS"; M=2; K=3; N=2; Act="relu6"; Tb=$WsTb }
)

foreach ($case in $MatmulActCases) {
    $tc = $case.Name
    Write-Host ""
    Write-Host "--- [generate matmul_act/$tc] ---" -ForegroundColor Cyan
    $genOut = & python "$TbDir\matmul\gen_matmul_data.py" --custom `
        --m $case.M --k $case.K --n $case.N `
        --dtype $case.Dtype --mode $case.Mode --bias --activation $case.Act --test-id $tc 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [GEN FAIL] $genOut" -ForegroundColor Red
        $TotalFail++
        $Results += [PSCustomObject]@{ Name="matmul_act/$tc"; Status="GEN FAIL"; Pass=0; Fail=1 }
        continue
    }

    $tcDir = "$TbDir\matmul\$tc"
    Run-Test -Name "matmul_act/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @($case.Tb)
}

$ConvActCases = @(
    @{ Name="conv2d_relu_otf_int8_os_default"; Dtype="int8"; Mode="OS"; Act="relu"; Tb="$TbDir\matmul\tb_matmul_os.v" },
    @{ Name="conv2d_relu6_otf_int8_ws_default"; Dtype="int8"; Mode="WS"; Act="relu6"; Tb=$WsTb },
    @{ Name="conv2d_relu6_otf_fp16_os_default"; Dtype="fp16"; Mode="OS"; Act="relu6"; Tb="$TbDir\matmul\tb_matmul_os.v" }
)

foreach ($case in $ConvActCases) {
    $tc = $case.Name
    Write-Host ""
    Write-Host "--- [generate conv2d_act_otf/$tc] ---" -ForegroundColor Cyan
    $genOut = & python "$ConvDir\gen_conv2d_im2col_data.py" `
        --on-the-fly --bias --activation $case.Act --dtype $case.Dtype --mode $case.Mode --test-id $tc 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [GEN FAIL] $genOut" -ForegroundColor Red
        $TotalFail++
        $Results += [PSCustomObject]@{ Name="conv2d_act_otf/$tc"; Status="GEN FAIL"; Pass=0; Fail=1 }
        continue
    }

    $tcDir = "$ConvDir\$tc"
    Run-Test -Name "conv2d_act_otf/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @($case.Tb)
}

# ---- 9. T6.5 INT8 quant/saturate tests ----
# Direct scalar postprocess order is accumulator -> optional bias -> activation
# -> optional INT8 quantize/saturate. Quant writes sign-extended int8 words.
$MatmulQuantCases = @(
    @{ Name="matmul_quant_int8_os_3x5x4"; Mode="OS"; M=3; K=5; N=4; Scale=3; Shift=5; Round=$true;  Act="relu"; Tb="$TbDir\matmul\tb_matmul_os.v" },
    @{ Name="matmul_quant_int8_ws_3x5x4"; Mode="WS"; M=3; K=5; N=4; Scale=1; Shift=3; Round=$false; Act="none"; Tb=$WsTb }
)

foreach ($case in $MatmulQuantCases) {
    $tc = $case.Name
    Write-Host ""
    Write-Host "--- [generate matmul_quant/$tc] ---" -ForegroundColor Cyan
    $roundArg = @()
    if ($case.Round) { $roundArg = @("--quant-round") }
    $genOut = & python "$TbDir\matmul\gen_matmul_data.py" --custom `
        --m $case.M --k $case.K --n $case.N `
        --dtype int8 --mode $case.Mode --bias --activation $case.Act `
        --quant --quant-scale $case.Scale --quant-shift $case.Shift @roundArg --test-id $tc 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [GEN FAIL] $genOut" -ForegroundColor Red
        $TotalFail++
        $Results += [PSCustomObject]@{ Name="matmul_quant/$tc"; Status="GEN FAIL"; Pass=0; Fail=1 }
        continue
    }

    $tcDir = "$TbDir\matmul\$tc"
    Run-Test -Name "matmul_quant/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @($case.Tb)
}

$ConvQuantCases = @(
    @{ Name="conv2d_quant_otf_int8_os_default"; Mode="OS"; Scale=2; Shift=3; Round=$true;  Act="relu"; Tb="$TbDir\matmul\tb_matmul_os.v" },
    @{ Name="conv2d_quant_otf_int8_ws_default"; Mode="WS"; Scale=1; Shift=2; Round=$false; Act="none"; Tb=$WsTb }
)

foreach ($case in $ConvQuantCases) {
    $tc = $case.Name
    Write-Host ""
    Write-Host "--- [generate conv2d_quant_otf/$tc] ---" -ForegroundColor Cyan
    $roundArg = @()
    if ($case.Round) { $roundArg = @("--quant-round") }
    $genOut = & python "$ConvDir\gen_conv2d_im2col_data.py" `
        --on-the-fly --bias --activation $case.Act --dtype int8 --mode $case.Mode `
        --quant --quant-scale $case.Scale --quant-shift $case.Shift @roundArg --test-id $tc 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [GEN FAIL] $genOut" -ForegroundColor Red
        $TotalFail++
        $Results += [PSCustomObject]@{ Name="conv2d_quant_otf/$tc"; Status="GEN FAIL"; Pass=0; Fail=1 }
        continue
    }

    $tcDir = "$ConvDir\$tc"
    Run-Test -Name "conv2d_quant_otf/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @($case.Tb)
}

# ---- 10. T6.6 Two-layer Conv2D end-to-end test ----
# Layer0 writes quantized OFM to DRAM; layer1 consumes the same address as its
# A matrix and the testbench checks both the intermediate and final outputs.
$ConvTwoLayerCases = @(
    @{ Name="conv2d_two_layer_int8_os_default"; Tb="$ConvDir\tb_conv2d_two_layer.v" }
)

foreach ($case in $ConvTwoLayerCases) {
    $tc = $case.Name
    Write-Host ""
    Write-Host "--- [generate conv2d_two_layer/$tc] ---" -ForegroundColor Cyan
    $genOut = & python "$ConvDir\gen_conv2d_two_layer_data.py" --test-id $tc 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [GEN FAIL] $genOut" -ForegroundColor Red
        $TotalFail++
        $Results += [PSCustomObject]@{ Name="conv2d_two_layer/$tc"; Status="GEN FAIL"; Pass=0; Fail=1 }
        continue
    }

    $tcDir = "$ConvDir\$tc"
    Run-Test -Name "conv2d_two_layer/$tc" `
        -VvpOut "$SimDir\reg_${tc}.vvp" `
        -IncDir $tcDir `
        -RunDir $tcDir `
        -ExtraSrc @($case.Tb)
}

# ---- Summary ----
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  REGRESSION SUMMARY" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

$width = ($Results | ForEach-Object { $_.Name.Length } | Measure-Object -Maximum).Maximum + 2
foreach ($r in $Results) {
    $padded = $r.Name.PadRight($width)
    $color  = if ($r.Status -eq "PASS") { "Green" } elseif ($r.Status -eq "COMPILE FAIL") { "Magenta" } else { "Red" }
    Write-Host ("  {0}  {1,4} PASS  {2,2} FAIL  [{3}]" -f $padded, $r.Pass, $r.Fail, $r.Status) -ForegroundColor $color
}

Write-Host ""
$overallColor = if ($TotalFail -eq 0) { "Green" } else { "Red" }
Write-Host ("  TOTAL: {0} PASS, {1} FAIL" -f $TotalPass, $TotalFail) -ForegroundColor $overallColor
Write-Host "======================================================" -ForegroundColor Cyan

if ($TotalFail -gt 0) { exit 1 }
