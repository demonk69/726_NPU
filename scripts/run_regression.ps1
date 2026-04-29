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

    Push-Location $ProjectRoot
    $simOut = & vvp $VvpOut 2>&1
    Pop-Location

    $pass = ($simOut | Select-String "\[PASS\]").Count
    $fail = ($simOut | Select-String "\[FAIL\]").Count
    $timeout = ($simOut | Select-String "TIMEOUT").Count

    # Also check summary lines (matmul testbench style: "ALL N CHECKS PASSED")
    $summaryLine = ($simOut | Select-String "RESULT:|ALL.*PASS|ALL.*CHECKS PASSED|PASSED.*FAILED" | Select-Object -Last 1)
    if ($summaryLine) { Write-Host "  $($summaryLine.Line.Trim())" -ForegroundColor Cyan }

    # Extract pass count from "ALL N CHECKS PASSED" if no [PASS] tags
    if ($pass -eq 0 -and $fail -eq 0) {
        $chkMatch = ($simOut | Select-String "ALL (\d+) CHECKS PASSED" | Select-Object -Last 1)
        if ($chkMatch) { $pass = [int]$chkMatch.Matches[0].Groups[1].Value }
        # Also handle "N PASSED, 0 FAILED" style
        $r2Match = ($simOut | Select-String "(\d+) PASSED.*?(\d+) FAILED" | Select-Object -Last 1)
        if ($r2Match -and $pass -eq 0) {
            $pass = [int]$r2Match.Matches[0].Groups[1].Value
            $fail = [int]$r2Match.Matches[0].Groups[2].Value
        }
    }

    if ($timeout -gt 0) {
        $fail += $timeout
        Write-Host "  [TIMEOUT detected]" -ForegroundColor Red
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
        -ExtraSrc @($WsTb)
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
