@echo off
setlocal

set TB=D:\NPU_prj\tb\matmul\tb_matmul_os.v
set RTL=D:\NPU_prj\rtl

set RTL_FILES=%RTL%\pe\fp16_mul.v %RTL%\pe\fp16_add.v %RTL%\pe\fp32_add.v %RTL%\pe\pe_top.v %RTL%\common\fifo.v %RTL%\common\axi_monitor.v %RTL%\common\op_counter.v %RTL%\array\pe_array.v %RTL%\buf\pingpong_buf.v %RTL%\power\npu_power.v %RTL%\ctrl\npu_ctrl.v %RTL%\axi\npu_axi_lite.v %RTL%\axi\npu_dma.v %RTL%\top\npu_top.v

set PASS=0
set FAIL=0

call :run_test "WS SQ INT8 4x4"   D:\NPU_prj\tb\matmul\ws_sq_int8_4x4
call :run_test "WS SQ INT8 8x8"   D:\NPU_prj\tb\matmul\ws_sq_int8_8x8
call :run_test "WS SQ INT8 16x16" D:\NPU_prj\tb\matmul\ws_sq_int8_16x16
call :run_test "WS SQ FP16 4x4"   D:\NPU_prj\tb\matmul\ws_sq_fp16_4x4
call :run_test "WS SQ FP16 8x8"   D:\NPU_prj\tb\matmul\ws_sq_fp16_8x8

echo.
echo ===============================
echo SUMMARY: %PASS% PASSED, %FAIL% FAILED
echo ===============================
goto :eof

:run_test
echo ===============================
echo %~1
echo ===============================
cd /d %~2
iverilog -o tb_matmul_os.vvp -I. %RTL_FILES% %TB% 2>nul
if errorlevel 1 (echo COMPILE FAILED & set /a FAIL+=1 & goto :eof)
vvp tb_matmul_os.vvp 2>&1 | findstr /C:"PASSED" /C:"FAILED"
goto :eof
