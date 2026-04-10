@echo off
setlocal enabledelayedexpansion

set ROOT=D:\NPU_prj
set RTL=%ROOT%\rtl\pe\fp16_mul.v %ROOT%\rtl\pe\fp16_add.v %ROOT%\rtl\pe\fp32_add.v %ROOT%\rtl\pe\pe_top.v %ROOT%\rtl\common\fifo.v %ROOT%\rtl\common\axi_monitor.v %ROOT%\rtl\common\op_counter.v %ROOT%\rtl\array\pe_array.v %ROOT%\rtl\buf\pingpong_buf.v %ROOT%\rtl\power\npu_power.v %ROOT%\rtl\ctrl\npu_ctrl.v %ROOT%\rtl\axi\npu_axi_lite.v %ROOT%\rtl\axi\npu_dma.v %ROOT%\rtl\top\npu_top.v
set TB=%ROOT%\tb\matmul\tb_matmul_os.v
set DIR=%ROOT%\tb\matmul

set TESTS=sq_int8_4x4 sq_int8_8x8 sq_int8_16x16 sq_fp16_4x4 sq_fp16_8x8

set TOTAL_PASS=0
set TOTAL_FAIL=0

echo ========================================
echo   Square Matrix Multiplication Tests
echo ========================================
echo.

for %%T in (%TESTS%) do (
    echo --- [%%T] ---
    pushd %DIR%\%%T
    iverilog -o tb_matmul_os.vvp -I. %RTL% %TB% > compile.log 2>&1
    if errorlevel 1 (
        echo   COMPILE ERROR
        type compile.log
    ) else (
        vvp tb_matmul_os.vvp > sim.log 2>&1
        if errorlevel 1 (
            echo   SIM ERROR
            type sim.log
        ) else (
            findstr /C:"ALL" /C:"PASSED" /C:"FAILED" sim.log
        )
    )
    popd
    echo.
)

echo ========================================
echo   Done
echo ========================================
