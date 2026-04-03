# =============================================================================
# run_soc_sim.ps1 - SoC simulation runner
#
# Prerequisites:
#   - Icarus Verilog (iverilog) installed
#   - RISC-V toolchain (riscv32-unknown-elf-as/objcopy) for firmware
#     OR use the pre-generated hex file
#
# Usage:
#   .\scripts\run_soc_sim.ps1
#
# Output:
#   soc_sim.vcd  - Waveform file (open with GTKWave)
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RtlDir = Join-Path $ProjectRoot "rtl"
$TbDir = Join-Path $ProjectRoot "tb"
$SimDir = Join-Path $ProjectRoot "sim"
$Picorv32Dir = Join-Path $ProjectRoot "picorv32_ref"

# Ensure sim directory exists
if (-not (Test-Path $SimDir)) { New-Item -ItemType Directory -Path $SimDir | Out-Null }

# Copy picorv32.v to sim directory
Copy-Item (Join-Path $Picorv32Dir "picorv32.v") (Join-Path $SimDir "picorv32.v") -Force

Write-Host "=== NPU SoC Simulation ===" -ForegroundColor Cyan
Write-Host ""

# ---- Step 1: Assemble firmware ----
Write-Host "[1/3] Assembling firmware..." -ForegroundColor Yellow
$riscv_as = Get-Command "riscv32-unknown-elf-as" -ErrorAction SilentlyContinue
if (-not $riscv_as) {
    $riscv_as = Get-Command "riscv64-unknown-elf-as" -ErrorAction SilentlyContinue
}
if (-not $riscv_as) {
    Write-Host "  WARNING: RISC-V assembler not found. Using pre-generated firmware hex." -ForegroundColor Red
    Write-Host "  Install riscv32-unknown-elf toolchain for full flow." -ForegroundColor DarkGray
    # Generate a minimal firmware hex that just loops
    # (This is a fallback - the real test needs the assembled firmware)
    if (-not (Test-Path (Join-Path $TbDir "soc_test.hex"))) {
        Write-Host "  ERROR: soc_test.hex not found. Cannot proceed." -ForegroundColor Red
        exit 1
    }
} else {
    $asmFile = Join-Path $TbDir "soc_test.S"
    $objFile = Join-Path $SimDir "soc_test.o"
    $elfFile = Join-Path $SimDir "soc_test.elf"
    $hexFile = Join-Path $TbDir "soc_test.hex"
    $binFile = Join-Path $SimDir "soc_test.bin"

    & $riscv_as.Source -march=rv32imc -o $objFile $asmFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Assembly failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "  OK: Assembled to $objFile" -ForegroundColor Green

    # Convert to binary
    $riscv_objcopy = $riscv_as.Source -replace "as$", "objcopy"
    & $riscv_objcopy -O binary $objFile $binFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: objcopy failed" -ForegroundColor Red
        exit 1
    }

    # Convert binary to hex (32-bit words)
    & iverilog -o NUL -E `
        -DDUMP_HEX="$binFile" `
        (Join-Path $RtlDir "soc\soc_top.v")
    # Fallback: use Python or custom script
    python3 -c "
import struct, sys
with open('$binFile', 'rb') as f:
    data = f.read()
# Pad to word boundary
while len(data) % 4 != 0:
    data += b'\x00'
with open('$hexFile', 'w') as f:
    for i in range(0, len(data), 4):
        word = struct.unpack('<I', data[i:i+4])[0]
        f.write(f'{word:08x}\n')
" 2>$null
    if ($LASTEXITCODE -ne 0) {
        # Try python
        python -c "
import struct, sys
with open('$binFile', 'rb') as f:
    data = f.read()
while len(data) % 4 != 0:
    data += b'\x00'
with open('$hexFile', 'w') as f:
    for i in range(0, len(data), 4):
        word = struct.unpack('<I', data[i:i+4])[0]
        f.write(f'{word:08x}\n')
"
    }
    Write-Host "  OK: Generated $hexFile" -ForegroundColor Green
}

# ---- Step 2: Compile Verilog ----
Write-Host "[2/3] Compiling Verilog..." -ForegroundColor Yellow

$srcFiles = @(
    # PicoRV32 CPU
    (Join-Path $SimDir "picorv32.v")
    # SoC modules
    (Join-Path $RtlDir "soc\soc_mem.v")
    (Join-Path $RtlDir "soc\dram_model.v")
    (Join-Path $RtlDir "soc\axi_lite_bridge.v")
    (Join-Path $RtlDir "soc\soc_top.v")
    # NPU modules
    (Join-Path $RtlDir "axi\npu_axi_lite.v")
    (Join-Path $RtlDir "ctrl\npu_ctrl.v")
    (Join-Path $RtlDir "axi\npu_dma.v")
    (Join-Path $RtlDir "buf\pingpong_buf.v")
    (Join-Path $RtlDir "common\fifo.v")
    (Join-Path $RtlDir "pe\pe_top.v")
    (Join-Path $RtlDir "pe\fp16_mul.v")
    (Join-Path $RtlDir "array\pe_array.v")
    (Join-Path $RtlDir "power\npu_power.v")
    (Join-Path $RtlDir "top\npu_top.v")
    # Testbench
    (Join-Path $TbDir "tb_soc.v")
)

$vvpFile = Join-Path $SimDir "soc_sim.vvp"
$arguments = "-o", $vvpFile, "-s", "tb_soc"
foreach ($f in $srcFiles) {
    if (Test-Path $f) {
        $arguments += $f
    } else {
        Write-Host "  WARNING: File not found: $f" -ForegroundColor DarkYellow
    }
}

& iverilog @arguments
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Compilation failed" -ForegroundColor Red
    exit 1
}
Write-Host "  OK: Compiled to $vvpFile" -ForegroundColor Green

# ---- Step 3: Run simulation ----
Write-Host "[3/3] Running simulation..." -ForegroundColor Yellow

Push-Location $SimDir
& vvp -N $vvpFile
$simExit = $LASTEXITCODE
Pop-Location

$vcdFile = Join-Path $SimDir "soc_sim.vcd"
if (Test-Path $vcdFile) {
    Write-Host "  OK: Waveform saved to $vcdFile" -ForegroundColor Green
    Write-Host "  View with: gtkwave $vcdFile" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
exit $simExit
